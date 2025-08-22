import torch
from torch_geometric.data import Data
import numpy as np

import os
import re

class LineReader:
    def __init__(self, file):
        self.iterator = iter(file)
        self._buffer = None
    def __iter__(self): return self
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

    def pre_process_all(self, input_dir, output_dir):
        graph_sequences = []
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if os.path.exists(input_dir):
            for file in os.listdir(input_dir):
                if file.endswith(".txt"):
                    simulation_id = re.search(r"-(\d+)\.txt", file)
                    simulation_id = simulation_id.group(1) if simulation_id else "0"

                    full_path = os.path.join(input_dir, file)
                    simulation_graphs = self.pre_process(full_path)
                    if simulation_graphs:
                        graph_sequences.append(simulation_graphs)
                        output_file = os.path.join(output_dir, "graph_" + simulation_id + ".pt")
                        torch.save(simulation_graphs, output_file)
        print(f"Total number of graph sequences: {len(graph_sequences)}")
        bbdd_file = os.path.join(input_dir, "GRAPHS.pt")
        print(f"Writing database to {bbdd_file}")
        torch.save(graph_sequences, bbdd_file)

    def pre_process(self, input_file: str):
        file_name = os.path.basename(input_file)
        print(f"PreProcessing {file_name}...")
        # simulation_id = re.search(r"-(\d+)\.txt", file_name)
        # simulation_id = int(simulation_id.group(1)) if simulation_id else 0
        step_graphs = self.__preprocess_input(input_file)
        return step_graphs

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

    # -----------------------------
    # Parsing del fichero
    # -----------------------------
    def __preprocess_input(self, input_file: str):
        file_name = os.path.basename(input_file)
        simulation_id = re.search(r"-(\d+)\.txt", file_name)
        simulation_id = simulation_id.group(1) if simulation_id else "0"

        nodal_positions_per_state = {}   # state -> {old_id: [x,y,z]}
        elements_nodes = []              # lista de elementos, cada uno = [old_id_i,...]
        with open(input_file, 'r') as input_f:
            iterator = LineReader(input_f)
            current_state = -1
            for line in iterator:
                if line.startswith("*ELEMENT") and not elements_nodes:
                    # leemos elementos como listas de old_ids (no edges aún)
                    elems, next_line = self.__read_elements(iterator)
                    elements_nodes.extend(elems)
                    if next_line:
                        iterator.push_back(next_line)
                elif line.startswith("$STATE_NO = "):
                    current_state += 1
                elif line.startswith("*NODE"):
                    nodes, next_line = self.__process_nodes(iterator)
                    nodal_positions_per_state[current_state] = nodes
                    if next_line:
                        iterator.push_back(next_line)

        # --- construir coords y orden de old_ids por fila ---
        coords, old_id_order = self.__prepare_nodes_inputs(nodal_positions_per_state)

        # --- mapping old_id -> new_idx (0..N-1) ---
        old2new, new2old = self._build_id_map(old_id_order)

        # --- reordenar coords a 0..N-1 (por si old_id_order no está compactado) ---
        idx_newpos = torch.tensor([old2new[int(oid)] for oid in old_id_order], dtype=torch.long)
        coords = coords[:, idx_newpos, :]  # (T,N,3) reordenado a 0..N-1

        # --- edge_index desde elementos remapeados ---
        edge_index = self._build_edge_index_from_elements(elements_nodes, old2new, bidirectional=True, clique=True)

        T, N, _ = coords.shape

        # ----------  detectar nodos fijos ----------
        bc_mask, fixed_idx, max_move = self.__infer_fixed_nodes(coords, 1e-6)
        num_fixed = int(bc_mask.sum())
        # print(f"[BC] fixed_tol={1e-6:g} → fixed nodes: {num_fixed}/{N}")

        # --- generar Data por timestep ---
        displacements = coords - coords[0]  # (T,N,3)
        pos0 = coords[0]
        data_list = []
        for t in range(T-1):
            x_t = displacements[t]
            y_t = displacements[t+1]
            edge_attr = self.__build_edge_attr(edge_index, pos0)
            data = Data(
                x=x_t,
                y=y_t,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos0=pos0,
                bc_mask = bc_mask,
                fixed_idx = fixed_idx,
                simulation_id = simulation_id
            )
            # guarda también el mapeo si te sirve para trazar resultados
            data.orig_node_id = torch.tensor(new2old, dtype=torch.long)  # tamaño N
            data.t_idx = torch.tensor([t], dtype=torch.long)

            # bc_feat = bc_mask.float().unsqueeze(1)     # (N,1)
            # data.x = torch.cat([data.x, bc_feat], dim=1)  # → in_channels = 4
            data_list.append(data)

        # sanity checks
        assert int(edge_index.max()) < N and int(edge_index.min()) >= 0, \
            "edge_index fuera de rango tras remapeo"
        return data_list
    
    def __infer_fixed_nodes(self, coords: torch.Tensor, tol: float):
        """
        coords: Tensor (T, N, 3) con coordenadas absolutas.
        Devuelve:
          - bc_mask: BoolTensor (N,) True si el nodo no se mueve (<= tol en toda la secuencia)
          - fixed_idx: LongTensor (K,) índices de nodos fijos
          - max_move: Tensor (N,) desplazamiento máximo por nodo (útil para depurar)
        """
        # Desplazamientos relativos al estado inicial
        disp = coords - coords[0]               # (T, N, 3)
        # Norma por timestep y nodo
        step_norm = torch.linalg.norm(disp, dim=2)   # (T, N)
        # Máximo a lo largo del tiempo
        max_move = step_norm.max(dim=0).values       # (N,)
        bc_mask = max_move <= tol
        fixed_idx = torch.nonzero(bc_mask, as_tuple=True)[0]
        return bc_mask, fixed_idx, max_move

    def __prepare_nodes_inputs(self, nodal_positions_per_state):
        timesteps = sorted(nodal_positions_per_state.keys())
        # conjunto de IDs presentes en el primer estado (asumimos mismos nodos en todos)
        node_ids = sorted(nodal_positions_per_state[timesteps[0]].keys())
        coords = np.array([
            [nodal_positions_per_state[t][n] for n in node_ids]
            for t in timesteps
        ], dtype=np.float32)
        coords = torch.tensor(coords, dtype=torch.float32)  # (T,N,3)
        return coords, node_ids
    
    def __build_edge_attr(self, edge_index, pos0):
        src, dst = edge_index
        rel_vec = pos0[dst] - pos0[src]            # (E, 3)
        length = rel_vec.norm(dim=-1, keepdim=True) # (E, 1)
        direction = rel_vec / (length + 1e-9)       # (E, 3)
        edge_attr = torch.cat([length, direction], dim=-1)  # (E, 4)
        return edge_attr

    def __read_elements(self, iterator):
        """
        Lee bloque *ELEMENT y devuelve lista de elementos como listas de old_ids (nodos).
        No construye aristas aquí; se hace después con el mapping old->new.
        """
        elements = []
        next_line = None
        for line in iterator:
            if line.startswith("*"):
                next_line = line
                break
            parts = line.strip().split()
            # formato típico: <elem_id> <part_id> n1 n2 n3 n4 ...
            if len(parts) < 3:
                continue
            # ignoramos elem_id y part_id; nos quedamos con nodos
            _, _, *nodes = parts
            try:
                nodes = [int(n) for n in nodes]
            except ValueError:
                continue
            elements.append(nodes)
        return elements, next_line

    def __process_nodes(self, iterator):
        nodes = {}
        next_line = None
        for line in iterator:
            if line.startswith("*"):
                next_line = line
                break
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            idx = int(parts[0])                     # old_id de nodo
            coords = list(map(float, parts[1:4]))
            nodes[idx] = coords
        return nodes, next_line

if __name__ == "__main__":
    preprocessing = DataPreProcessing()
    input_dir = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/simulation_results/clean_results/"
    output_dir = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/simulation_results/graphs/"
    preprocessing.pre_process_all(input_dir, output_dir)
