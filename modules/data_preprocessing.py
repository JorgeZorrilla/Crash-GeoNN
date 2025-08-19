import torch
from torch_geometric.data import Data
import numpy as np

import os
import re

class LineReader:
    def __init__(self, file):
        self.iterator = iter(file)
        self._buffer = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._buffer is not None:
            line = self._buffer
            self._buffer = None
            return line
        return next(self.iterator)

    def push_back(self, line):
        self._buffer = line

class DataPreProcessing:
    def __init__(self) -> None:
        pass

       # -----------------------------
    # Helpers de normalización IDs
    # -----------------------------
    @staticmethod
    def _build_id_map(node_ids):
        """
        node_ids: iterable de IDs originales presentes en la simulación.
        return:
          old2new: dict old_id -> new_idx (0..N-1)
          new2old: list con el old_id de cada fila 0..N-1
        """
        node_ids_sorted = sorted(set(int(n) for n in node_ids))
        old2new = {old: i for i, old in enumerate(node_ids_sorted)}
        new2old = node_ids_sorted
        return old2new, new2old

    @staticmethod
    def _build_edge_index_from_elements(elements, old2new, bidirectional=True, clique=True):
        """
        elements: lista de elementos, cada uno = [old_id_i, old_id_j, ...]
        old2new: dict old -> new
        bidirectional: duplica aristas (i,j) y (j,i)
        clique: si True conecta todos los pares del elemento (más denso y suele ir mejor).
                si False conecta solo pares consecutivos (anillo).
        """
        undirected_pairs = set()
        for elem in elements:
            ids = [old2new[int(u)] for u in elem if int(u) in old2new]
            L = len(ids)
            if L < 2:
                continue
            if clique:
                for a in range(L):
                    for b in range(a+1, L):
                        u, v = ids[a], ids[b]
                        if u == v: continue
                        undirected_pairs.add((u, v) if u < v else (v, u))
            else:
                for a in range(L):
                    u, v = ids[a], ids[(a+1) % L]
                    if u == v: continue
                    undirected_pairs.add((u, v) if u < v else (v, u))

        edges = []
        if bidirectional:
            for u, v in undirected_pairs:
                edges.append([u, v])
                edges.append([v, u])
        else:
            edges = [[u, v] for (u, v) in undirected_pairs]

        if len(edges) == 0:
            raise ValueError("No edges built from elements; check input.")
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def pre_process_all(self, input_dir):
        graph_sequences = []
        if os.path.exists(input_dir):
            files = os.listdir(input_dir)
            for file in files:
                if file.endswith(".txt"):
                    full_path = os.path.join(input_dir, file)
                    simulation_graphs =  self.pre_process(full_path)
                    if simulation_graphs:
                        # graph_sequences = graph_sequences + simulation_graphs
                        graph_sequences.append(simulation_graphs)
        print(f"Total number of graph sequences: {len(graph_sequences)}")
        bbdd_file = input_dir + "/GRAPHS.pt"
        print(f"Writing database to {bbdd_file}")
        torch.save(graph_sequences, bbdd_file)
        

    def pre_process(self, input_file: str):
        file_name = os.path.basename(input_file)
        print(f"PostProcessing {file_name}...")
        simulation_id = re.search(r"-(\d+)\.txt", file_name)
        simulation_id = int(simulation_id.group(1)) if simulation_id else 0
            
        step_graphs = self.__preprocess_input(input_file)
        return step_graphs

    
    def __preprocess_input(self, input_file: str):
        nodals_positions_per_state = {}
        connectivity = None

        with open(input_file, 'r') as input:
            iterator = LineReader(input)
            current_state = -1
            for line in iterator:
                if line.startswith("*ELEMENT") and connectivity is None:
                    connectivity, next_line = self.__process_connectivity(iterator)
                    if next_line:
                        iterator.push_back(next_line)
                elif line.startswith("$STATE_NO = "):
                    current_state += 1
                elif line.startswith("*NODE"):
                    nodes, next_line = self.__process_nodes(iterator)
                    nodals_positions_per_state[current_state] = nodes
                    if next_line:
                        iterator.push_back(next_line)
        coords = self.__prepare_nodes_inputs(nodals_positions_per_state)

        T, num_nodes, dim = coords.shape

        # # Initial node features
        # x = coords[0] # (num_nodes, 3)
        # # Target: relative displacements in time
        # displacements = coords - coords[0]   # (T, num_nodes, 3)

        # data = Data(
        #     x = x,
        #     edge_index= connectivity,
        #     y = displacements
        # )

        displacements = coords - coords[0]   # (T, num_nodes, 3)
        pos0 = coords[0]
        data_list = []
        for t in range(T-1):  # hasta T-2 porque predices t+1
            x_t = displacements[t]           # (num_nodes, 3)
            y_t = displacements[t+1]         # (num_nodes, 3)
            
            data = Data(
                x=x_t, 
                edge_index=connectivity, 
                y=y_t,
                pos0=pos0
            )
            data_list.append(data)

        return data_list
        # return connectivity, nodals_positions_per_state
    
    def __prepare_nodes_inputs(self, nodal_positions_per_state):
        timesteps = sorted(nodal_positions_per_state.keys())
        node_ids = sorted(next(iter(nodal_positions_per_state.values())).keys())

        coords = np.array([
            [nodal_positions_per_state[t][n] for n in node_ids]   # nodos por timestep
            for t in timesteps                   # recorrer timesteps
        ])

        if not torch.is_tensor(coords):
            coords = torch.tensor(coords, dtype=torch.float)
        return coords

   
    def __process_connectivity(self, iterator):
        edges = set()
        line = None
        for line in iterator:
            if line.startswith("*"):
                break
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            _, _, *nodes = map(int, parts)
            n = len(nodes)
            for i in range(n):
                node1 = nodes[i]
                node2 = nodes[(i+1) % n]
                if node1 != node2:
                    edges.add((node1, node2))
                    edges.add((node2, node1))
        return torch.tensor(list(edges), dtype=torch.long).t().contiguous(), line
    
    def __process_nodes(self, iterator):
        nodes = {}
        for line in iterator:
            if line.startswith("*"):
                return nodes, line
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            idx = int(parts[0])
            coords = list(map(float, parts[1:4]))
            nodes[idx] = coords
        return nodes, None
    
if __name__ == "__main__":
    preprocessing = DataPreProcessing()
    # input_file = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/simulation_results/Geometry-093/output_data.txt"
    # postprocessing.post_process(input_file)

    input_dir = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/simulation_results/clean_results/"
    preprocessing.pre_process_all(input_dir)
    
