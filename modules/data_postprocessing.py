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

class DataPostProcessing:
    def __init__(self) -> None:
        pass

    def post_process_all(self, input_dir):
        graph_sequences = []
        if os.path.exists(input_dir):
            files = os.listdir(input_dir)
            for file in files:
                if file.endswith(".txt"):
                    full_path = os.path.join(input_dir, file)
                    simulation_graphs =  self.post_process2(full_path)
                    if simulation_graphs:
                        # graph_sequences = graph_sequences + simulation_graphs
                        graph_sequences.append(simulation_graphs)
        print(f"Total number of graph sequences: {len(graph_sequences)}")
        bbdd_file = input_dir + "/GRAPHS.pt"
        print(f"Writing database to {bbdd_file}")
        torch.save(graph_sequences, bbdd_file)
        

    def post_process2(self, input_file: str):
        file_name = os.path.basename(input_file)
        print(f"PostProcessing {file_name}...")
        simulation_id = re.search(r"-(\d+)\.txt", file_name)
        simulation_id = int(simulation_id.group(1)) if simulation_id else 0
            
        step_graphs = self.__process_input2(input_file)
        return step_graphs

        
    def post_process(self, input_file: str):
        file_name = os.path.basename(input_file)
        print(f"PostProcessing {file_name}...")
        simulation_id = re.search(r"-(\d+)\.txt", file_name)
        simulation_id = int(simulation_id.group(1)) if simulation_id else 0
            
        connectivity, nodal_positions_per_state = self.__process_input(input_file)
        if connectivity and nodal_positions_per_state:
            return self.__toGraph(simulation_id, connectivity, nodal_positions_per_state)
    
    def __toGraph(self, id, connectivity, nodal_positions_per_state):
        edge_index = torch.tensor(list(connectivity), dtype=torch.long).t().contiguous()

        graph_sequence = []

        for timestep in sorted(nodal_positions_per_state.keys()):
            node_dict = nodal_positions_per_state[timestep]

            # sort node by corresponding ID in edge_index
            sorted_node_ids = sorted(node_dict.keys())
            positions = [node_dict[nid] for nid in sorted_node_ids]
            x = torch.tensor(positions, dtype=torch.float)

            data = Data(
                x= x,
                edge_index = edge_index,
                sim_id = torch.tensor([id], dtype=torch.long),
                timestep = torch.tensor([timestep], dtype=torch.long)
            )

            graph_sequence.append(data)
        return graph_sequence
    
    def __toTorch(self, connectivity, nodal_positions_per_state):
        print("Preparing for torch")
        # Paso 1: obtener todos los nodos y su orden
        all_node_ids = sorted(list(next(iter(nodal_positions_per_state.values())).keys()))
        id_to_idx = {nid: idx for idx, nid in enumerate(all_node_ids)}

        # Paso 2: edge_index con indices consecutivos
        edge_index = torch.tensor([
            [id_to_idx[src] for (src, dst) in connectivity],
            [id_to_idx[dst] for (src, dst) in connectivity]
        ], dtype=torch.long)

        data_list = []

        # Paso 3: para cada timestep, crear un objeto Data
        for state, pos_dict in nodal_positions_per_state.items():
            x_list = [pos_dict[nid] for nid in all_node_ids]  # respetamos orden
            x = torch.tensor(x_list, dtype=torch.float32)  # shape [N, 3]

            data = Data(x=x, edge_index=edge_index)
            data.t = state  # opcional
            data_list.append(data)

        return data_list
    
    def __process_input2(self, input_file: str):
        nodals_positions_per_state = {}
        connectivity = None

        with open(input_file, 'r') as input:
            iterator = LineReader(input)
            current_state = -1
            for line in iterator:
                if line.startswith("*ELEMENT") and connectivity is None:
                    connectivity, next_line = self.__process_elements2(iterator)
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

    def __process_input(self, input_file: str):
        nodals_positions_per_state = {}
        connectivity = None

        with open(input_file, 'r') as input:
            iterator = LineReader(input)
            current_state = None
            for line in iterator:
                if line.startswith("*ELEMENT") and connectivity is None:
                    connectivity, next_line = self.__process_elements(iterator)
                    if next_line:
                        iterator.push_back(next_line)
                elif line.startswith("$STATE_NO = "):
                    current_state = int(line.split("=")[1])
                elif line.startswith("*NODE"):
                    nodes, next_line = self.__process_nodes(iterator)
                    nodals_positions_per_state[current_state] = nodes
                    if next_line:
                        iterator.push_back(next_line)
        return connectivity, nodals_positions_per_state
    
    def __create_connectivity(self, edge_index_list):
        return torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    
    def __process_elements2(self, iterator):
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
    
    def __process_elements(self, iterator):
        edges = set()
        for line in iterator:
            if line.startswith("*"):
                return edges, line
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            _, _, *nodes = map(int, parts)
            n = len(nodes)
            for i in range(n):
                node1 = nodes[i]
                node2 = nodes[(i+1) % n]
                if node1 != node2:
                    edge = tuple(sorted((node1, node2)))
                    edges.add(edge) 
        return edges, None

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
    postprocessing = DataPostProcessing()
    # input_file = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/simulation_results/Geometry-093/output_data.txt"
    # postprocessing.post_process(input_file)

    input_dir = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/simulation_results/clean_results/"
    postprocessing.post_process_all(input_dir)
    
