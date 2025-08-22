import numpy as np

import os
import re

from lasso.dyna import D3plot
from torch_geometric.data import Data
import torch

class D3PlotPostProcess:
    def __init__(self) -> None:
        pass

    def process_all(self, input_dir, output_dir):
        # graph_sequences = []
        graph_sequences = 0
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if os.path.exists(input_dir):
            available_directories = os.listdir(input_dir)
            for dir in available_directories:
                dir_path = os.path.join(input_dir, dir)
                if os.path.isdir(dir_path):
                    dir_files = os.listdir(dir_path)
                    if "d3plot" in dir_files:
                        simulation_id = re.search(r"Geometry-(\d+)", dir)
                        simulation_id = simulation_id.group(1) if simulation_id else "0"

                        full_path = os.path.join(dir_path, "d3plot")
                        print(f"Processing {full_path}...")
                        simulation_graphs = self.process(full_path, simulation_id)
                        if simulation_graphs:
                            graph_sequences += len(simulation_graphs)
                            # graph_sequences.append(simulation_graphs)
                            output_file = os.path.join(output_dir, "graph_" + simulation_id + ".pt")
                            print(f"Creating {output_file} to store the graphs")
                            torch.save(simulation_graphs, output_file)
                        print("Finished!")
        # print(f"Total number of graph sequences: {len(graph_sequences)}")
        print(f"Total number of graph sequences: {graph_sequences}")

    def process(self, input_path: str, simulation_id: str):
        if not os.path.exists(input_path):
            print(f"File {input_path} does not exist!")
            return
        d3 = D3plot(input_path)
        arrays = d3.arrays
        # print(arrays.keys())

        # Node fields
        nodes_id = arrays["node_ids"] 
        initial_nodes_coordinates = arrays["node_coordinates"]
        nodes_displacement = arrays["node_displacement"]
        nodes_coordinates = initial_nodes_coordinates[None, :, :] + nodes_displacement
        nodes_velocities = arrays["node_velocity"]
        nodes_acceleration = arrays["node_acceleration"]

        # Element fields
        elements_id = arrays["element_shell_ids"]
        elements_nodes_indexes = arrays["element_shell_node_indexes"]
        elements_parts = arrays["element_shell_part_indexes"]
        elements_stress = arrays["element_shell_stress"] # T, E, 3, 6
        elements_stress_mean = elements_stress.mean(axis=2) # T, E, 6
        elements_von_mises = self.__von_mises(elements_stress_mean) # T, E
        elements_strain = arrays["element_shell_effective_plastic_strain"] # T, E, 3
        elements_strain_mean = elements_strain.mean(axis=2) # T, E

        nodes_parts_indexes = self.__infer_nodes_solid(elements_parts, elements_nodes_indexes) # index = 0 RIGIDO, index = 1 B-Pillar
        nodes_rigid_mask = 1 - nodes_parts_indexes
        

        # Nodes tensors
        nodes_coords_tensor = torch.tensor(nodes_coordinates, dtype=torch.float32) # (T, N, 3)
        nodes_initial_coords_tensor = torch.tensor(initial_nodes_coordinates, dtype=torch.float32) # (T, N, 3)
        nodes_displacement_tensor = torch.tensor(nodes_displacement, dtype=torch.float32) # (T, N, 3)
        nodes_velocities_tensor = torch.tensor(nodes_velocities, dtype=torch.float32) # (T, N, 3)
        nodes_accelerations_tensor = torch.tensor(nodes_acceleration, dtype=torch.float32) # (T, N, 3)
        nodes_parts_indexes_tensor = torch.tensor(nodes_parts_indexes, dtype=torch.int) # (N)
        nodes_rigid_mask_tensor = torch.tensor(nodes_rigid_mask, dtype=torch.int) # (N)
        nodes_bc_mask_tensor, _, _ = self.__infer_fixed_nodes(nodes_coords_tensor, 1e-6) # (N))

        # Elements tensor
        edge_index_tensor, nodes_elements_index = self._build_and_adjacency_edge_index(elements_nodes_indexes, bidirectional=True, clique=False)
        elements_von_mises_tensor = torch.tensor(elements_von_mises, dtype=torch.float32)
        elements_strain_mean_tensor = torch.tensor(elements_strain_mean, dtype=torch.float32)

        T, N, _ = nodes_coords_tensor.shape

        pos0 = nodes_coords_tensor[0]
        data_list = []
        for t in range(T-1):
            coords_t = nodes_coords_tensor[t] # (N,3)
            coords_t1 = nodes_coords_tensor[t + 1] # (N,3)
            displacements_t = nodes_displacement_tensor[t]
            displacements_t1 = nodes_displacement_tensor[t+1]
            v_t = nodes_velocities_tensor[t] # (N,3)
            v_t1 = nodes_velocities_tensor[t + 1] # (N,3)
            a_t = nodes_accelerations_tensor[t] # (N,3)
            a_t1 = nodes_accelerations_tensor[t + 1] # (N,3)

            svm_t = self.__avg_element_field_to_node(nodes_elements_index, elements_von_mises_tensor[t])
            svm_t1 = self.__avg_element_field_to_node(nodes_elements_index, elements_von_mises_tensor[t+1])
            peeq_t = self.__avg_element_field_to_node(nodes_elements_index, elements_strain_mean_tensor[t])
            peeq_t1 = self.__avg_element_field_to_node(nodes_elements_index, elements_strain_mean_tensor[t+1])


            x_t = torch.cat([displacements_t, v_t, a_t, svm_t, peeq_t], dim=1) # N_CH = 11
            y_t = torch.cat([displacements_t1, v_t1, a_t1, svm_t1, peeq_t1], dim=1)  # N_CH = 11
            assert x_t.shape == y_t.shape

            edge_attr_tensor = self.__build_edge_attr(edge_index_tensor, pos0)
            data = Data(
                x=x_t,
                y=y_t,
                edge_index=edge_index_tensor,
                edge_attr=edge_attr_tensor,
                pos0=pos0,
                bc_mask = nodes_bc_mask_tensor,
                rigid_mask = nodes_rigid_mask_tensor,
                simulation_id = simulation_id,
                t_idx = torch.tensor([t], dtype=torch.long)
            )
        
            data_list.append(data)

        return data_list

    def __build_id_map(self, node_ids):
        """
        We need the max node id to be equal to the size of the nodes set to avoid problems later
        node_ids: Old Node IDS present in the simulation
        """
        node_ids_sorted = sorted(set(int(n) for n in node_ids))
        old2new = {old: i for i, old in enumerate(node_ids_sorted)}
        new2old = node_ids_sorted
        return old2new, new2old
    
    
    @staticmethod
    def _build_and_adjacency_edge_index(elements, bidirectional=True, clique=True):
        """
        elements: lista de elementos, cada uno = [old_id_i, old_id_j, ...]
        bidirectional: duplica aristas (i,j) y (j,i)
        clique: si True conecta todos los pares del elemento (más denso y suele ir mejor).
                si False conecta solo pares consecutivos (anillo).
        """
        nodes_elements_index = {}

        undirected_pairs = set()
        for i in range(len(elements)):
            ids = elements[i]

            for id in ids:
                if id not in nodes_elements_index:
                    nodes_elements_index[id] = set()
                nodes_elements_index[id].add(i)

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
        return edge_index, nodes_elements_index

    def __infer_nodes_solid(self, elements_parts, elements_nodes_index):
        assert len(elements_parts) == len(elements_nodes_index)
        N = np.max(elements_nodes_index) + 1
        nodes_parts_index = np.full(N, -1, dtype=int)

        for i in range(len(elements_parts)):
            part = elements_parts[i]
            affected_nodes_indexes = elements_nodes_index[i]
            for node_index in affected_nodes_indexes:
                nodes_parts_index[node_index] = part

        assert not np.any(nodes_parts_index == -1)
        return nodes_parts_index
    
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
    
    def __avg_element_field_to_node(self, nodes_elements_index, element_field):
        N = len(nodes_elements_index)
        node_field = np.full(N, -99999, dtype=int)
        for node_index, elements in nodes_elements_index.items():
            total_node_value = 0
            for element in elements:
                total_node_value += float(element_field[element])
            avg_node_value = total_node_value / len(elements)
            node_field[node_index] = avg_node_value
        assert not np.any(node_field == -99999)
        
        return torch.tensor(node_field, dtype=torch.float).float().unsqueeze(1)

    def __von_mises(self, sig6):
        """sig6: (..., 6) -> von Mises (broadcast). Orden [sxx, syy, szz, sxy, syz, szx]."""
        sxx = sig6[..., 0]; syy = sig6[..., 1]; szz = sig6[..., 2]
        sxy = sig6[..., 3]; syz = sig6[..., 4]; szx = sig6[..., 5]
        return np.sqrt(
            0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2) +
            3.0 * (sxy**2 + syz**2 + szx**2)
        )
    def __build_edge_attr(self, edge_index, pos0):
            src, dst = edge_index
            rel_vec = pos0[dst] - pos0[src]            # (E, 3)
            length = rel_vec.norm(dim=-1, keepdim=True) # (E, 1)
            direction = rel_vec / (length + 1e-9)       # (E, 3)
            edge_attr = torch.cat([length, direction], dim=-1)  # (E, 4)
            return edge_attr

    @staticmethod
    def elem_scalar_to_node_avg(elem_scalar, elements_nodes_idx, N):
        """
        elem_scalar: (Ne,) escalar por elemento (p.ej., von Mises ya promediado por capas)
        elements_nodes_idx: lista de listas con índices 0..N-1 de los nodos de cada elemento
        N: nº nodos
        -> Devuelve (N,) con promedio simple por nodo.
        """
        acc = np.zeros(N, dtype=np.float64)
        cnt = np.zeros(N, dtype=np.int32)
        for e, nodes_idx in enumerate(elements_nodes_idx):
            if not nodes_idx: 
                continue
            v = float(elem_scalar[e])
            for j in nodes_idx:
                acc[j] += v
                cnt[j] += 1
        cnt[cnt == 0] = 1
        return (acc / cnt).astype(np.float32)


if __name__ == "__main__":
    postprocess = D3PlotPostProcess()
    # input_file = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/d3plots/Geometry-0/d3plot"
    # postprocess.process(input_file, "0")
    input_dir = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/d3plots/"
    output_dir = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/d3plots/graphs"
    postprocess.process_all(input_dir, output_dir)