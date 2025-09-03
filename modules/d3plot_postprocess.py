import numpy as np

import os
import re

from lasso.dyna import D3plot
from torch_geometric.data import Data
import torch
import vtk
from vtk.util import numpy_support

class D3PlotPostProcess:
    def __init__(self) -> None:
        pass

    def process_all(self, input_dir, output_dir, time_step = 1):
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
                        simulation_graphs = self.process_with_vtk(full_path, simulation_id, time_step)
                        if simulation_graphs:
                            graph_sequences += len(simulation_graphs)
                            # graph_sequences.append(simulation_graphs)
                            output_file = os.path.join(output_dir, "graph_" + simulation_id + ".pt")
                            print(f"Creating {output_file} to store the graphs")
                            torch.save(simulation_graphs, output_file)
                        print("Finished!")
        # print(f"Total number of graph sequences: {len(graph_sequences)}")
        print(f"Total number of graph sequences: {graph_sequences}")
    
    def process_with_vtk(self, input_path: str, simulation_id: str, time_step = 1):
        if not os.path.exists(input_path):
            print(f"File {input_path} does not exist!")
            return
        
        time_steps, nodes_displacement, nodes_velocities, nodes_acceleration, pos0, nodes_rigid_mask, edge_index_tensor, nodes_elements_index, cells_von_mises, cells_pressures, cells_efp = self.__read_with_vtk(input_path)

        global_vel_results = self.check_vel_disp_consistency(nodes_displacement, nodes_velocities, time_steps, simulation_id)
        vel_scheme = global_vel_results['global']['best']
        nodes_displacement_tensor = torch.tensor(nodes_displacement, dtype=torch.float32) # (T, N, 3)
        nodes_velocities_tensor = torch.tensor(nodes_velocities, dtype=torch.float32) # (T, N, 3)
        nodes_accelerations_tensor = torch.tensor(nodes_acceleration, dtype=torch.float32) # (T, N, 3)
        pos0_tensor = torch.tensor(pos0, dtype=torch.float32) # (N, 3)


        nodes_rigid_mask_tensor = torch.tensor(nodes_rigid_mask, dtype=torch.int) # (N)
        fixed_ids, nodes_bc_mask = self.__find_static_nodes(nodes_displacement_tensor, 1e-6, True, True) # (N))
        nodes_bc_mask_tensor = torch.tensor(nodes_bc_mask, dtype=torch.int) # (N)

        T, N, _ = nodes_displacement_tensor.shape


        data_list = []
        for t in range(0, T-1, time_step):
            displacements_t = nodes_displacement_tensor[t]
            displacements_t1 = nodes_displacement_tensor[t+1]
            v_t = nodes_velocities_tensor[t] # (N,3)
            v_t1 = nodes_velocities_tensor[t + 1] # (N,3)
            a_t = nodes_accelerations_tensor[t] # (N,3)
            a_t1 = nodes_accelerations_tensor[t + 1] # (N,3)

            vm_node_t1 = self.__avg_element_field_to_node(nodes_elements_index, cells_von_mises[t+1], N)
            p_node_t1  = self.__avg_element_field_to_node(nodes_elements_index, cells_pressures[t+1], N)
            peeq_t1   = self.__avg_element_field_to_node(nodes_elements_index, cells_efp[t+1],       N)

            vm_node_t1 = torch.from_numpy(vm_node_t1).float().unsqueeze(1)   # (N,1)
            p_node_t1  = torch.from_numpy(p_node_t1 ).float().unsqueeze(1)   # (N,1)
            peeq_t1   = torch.from_numpy(peeq_t1).float().unsqueeze(1)    # (N,1)

            bc_col    = nodes_bc_mask_tensor.float().unsqueeze(1)     # (N,1)
            rigid_col = nodes_rigid_mask_tensor.float().unsqueeze(1)  # (N,1)

            edge_attr_tensor = self.__build_edge_attr(edge_index_tensor, pos0_tensor)

            dt_sum = np.sum(time_steps[t+1 : t+1+time_step])   # suma de los pasos intermedios
            t1_step = torch.tensor(float(dt_sum), dtype=torch.float32)
            data = None

            displacementFocused = True
            if displacementFocused:
                x_t = torch.cat([displacements_t,bc_col, rigid_col], dim=1) # N_CH = 5
                y_t = displacements_t1 # (N,3)
                phys_target_t1= torch.cat([vm_node_t1, p_node_t1, peeq_t1], dim=1)  # [N,3]

                data = Data(
                x=x_t, 
                y=y_t,
                edge_index=edge_index_tensor,
                edge_attr=edge_attr_tensor,  # (geom. de pos0)
                pos0=pos0_tensor,
                bc_mask=nodes_bc_mask_tensor,
                rigid_mask=nodes_rigid_mask_tensor,
                v_t = v_t,
                v_t1=v_t1,
                v_scheme = vel_scheme,
                aux_phys_target=phys_target_t1,
                simulation_id = simulation_id,
                t1_step = t1_step,
                t_idx=torch.tensor([t])
                )
            else:
                x_t = torch.cat([displacements_t, v_t, a_t,
                                bc_col, rigid_col], dim=1) # N_CH = 11
                y_t = torch.cat([displacements_t1, v_t1, a_t1], dim=1)  # N_CH = 9

                data = Data(
                    x=x_t,
                    y=y_t,
                    edge_index=edge_index_tensor,
                    edge_attr=edge_attr_tensor,
                    pos0=pos0_tensor,
                    bc_mask = nodes_bc_mask_tensor,
                    rigid_mask = nodes_rigid_mask_tensor,
                    simulation_id = simulation_id,
                    t_idx = torch.tensor([t], dtype=torch.long)
                )
        
            data_list.append(data)

        return data_list
    
    def __read_with_vtk(self, input_path):
        reader = vtk.vtkLSDynaReader()
        reader.SetFileName(input_path)
        reader.UpdateInformation()


        # Número de timesteps disponibles
        nsteps = reader.GetNumberOfTimeSteps()
        times = reader.GetNumberOfTimeSteps()
        reader.GetTimeStepRange()

        print("Time steps:", times)
        all_time_steps = []
        all_displacements = []
        all_velocities = []
        all_accelerations = []
        all_von_mises = []
        all_pressures = []
        all_efp = []
        pos0 = None
        connectivity = None
        nodes_elements_index = None
        rigid_mask = None
        time_step = reader.GetTimeValue(1) - reader.GetTimeValue(0)
        for t in range(nsteps):
            reader.SetTimeStep(t)
            reader.Update()
            mb = reader.GetOutput()
            n_blocks = mb.GetNumberOfBlocks()
            append = vtk.vtkAppendFilter()
            for i in range(n_blocks):
                append.AddInputData(mb.GetBlock(i))
            append.Update()

            time_step = reader.GetTimeValue(t) - reader.GetTimeValue(t-1) if t > 0 else 0
            all_time_steps.append(time_step)

            dataset = append.GetOutput()
            # print("Nº de nodos combinados:", dataset.GetNumberOfPoints())
            # print("Nº de elements combinados:", dataset.GetNumberOfCells())
            if isinstance(dataset, vtk.vtkUnstructuredGrid):
                pd = dataset.GetPointData()
                # print("  Arrays de nodos:", [pd.GetArrayName(j) for j in range(pd.GetNumberOfArrays())])
                displacements = numpy_support.vtk_to_numpy(pd.GetArray("Deflection")) # (N,3)
                all_displacements.append(displacements)
                velocities = numpy_support.vtk_to_numpy(pd.GetArray("Velocity")) # (N,3)
                all_velocities.append(velocities)
                accelerations = numpy_support.vtk_to_numpy(pd.GetArray("Acceleration")) # (N,3)
                all_accelerations.append(accelerations)
                absolute_coords = numpy_support.vtk_to_numpy(pd.GetArray("Deflected Coordinates"))  # (N,3)
                if t == 0 and pos0 == None:
                    pos0 = absolute_coords # (N,3)
                if rigid_mask == None:
                    nodes_ids =  numpy_support.vtk_to_numpy(dataset.GetPointData().GetArray("UserID"))
                    rigid_solid_ids = numpy_support.vtk_to_numpy(mb.GetBlock(0).GetPointData().GetArray("UserID"))
                    rigid_mask = [id in rigid_solid_ids for id in nodes_ids]
                
                cd = dataset.GetCellData()
                # print("Número de elementos:", dataset.GetNumberOfCells())
                # print("  Arrays de celdas:", [cd.GetArrayName(j) for j in range(cd.GetNumberOfArrays())])
                cells = dataset.GetCells()  # objeto vtkCellArray
                stress = numpy_support.vtk_to_numpy(cd.GetArray("Stress")) # (E,6) 
                xx, yy, zz, xy, yz, zx = [stress[:, i] for i in range(6)]
                all_von_mises.append(self.__compute_von_mises(xx, yy, zz, xy, yz, zx)) # (T, E, 6)
                all_pressures.append(self.__compute_pressure(xx, yy, zz)) # ()
                efp = numpy_support.vtk_to_numpy(cd.GetArray("EffPlastStrn")) # (E,1)
                all_efp.append(efp)
                

                if connectivity == None:
                    connectivity, nodes_elements_index = self._build_and_adjacency_edge_index_from_vtk(dataset)
                # # recorrer elementos
                # for i in range(dataset.GetNumberOfCells()):
                #     cell = dataset.GetCell(i)  # vtkCell (ej. vtkQuad, vtkHexahedron…)
                #     ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
                #     print("Elemento", i, "con nodos:", ids)

                    
        all_displacements = np.stack(all_displacements, axis=0) # (T,N,3)
        all_velocities = np.stack(all_velocities, axis=0) # (T,N,3)
        all_accelerations = np.stack(all_accelerations, axis=0) # (T,N,3)
        
        all_von_mises = np.stack(all_von_mises, axis=0) # (T,E)
        all_pressures = np.stack(all_pressures, axis=0) # (T,E)
        all_efp = np.stack(all_efp, axis=0) # (T,E)


        return all_time_steps, all_displacements, all_velocities, all_accelerations, pos0, rigid_mask, connectivity, nodes_elements_index, all_von_mises, all_pressures, all_efp

    
    def __compute_von_mises(self, xx, yy, zz, xy, yz, zx):
        vm = 0.5*((xx-yy)**2 + (yy-zz)**2 + (zz-xx)**2) + 3*(xy**2 + yz**2 + zx**2)
        vm = np.sqrt(vm)
        return vm
    
    def __compute_pressure(self, xx, yy, zz):
        return -(xx + yy + zz) / 3.0

    def __build_id_map(self, node_ids):
        """
        We need the max node id to be equal to the size of the nodes set to avoid problems later
        node_ids: Old Node IDS present in the simulation
        """
        node_ids_sorted = sorted(set(int(n) for n in node_ids))
        old2new = {old: i for i, old in enumerate(node_ids_sorted)}
        new2old = node_ids_sorted
        return old2new, new2old
    
    def __find_static_nodes(self, disp, tol: float = 1e-9, ignore_nan: bool = True, return_mask: bool = False):
        """
        Devuelve los índices de nodos que NUNCA se mueven más de 'tol'
        a lo largo de todos los timesteps.

        Parámetros
        ----------
        disp : np.ndarray | torch.Tensor, shape (T, N, 3)
            Desplazamientos acumulados respecto a la posición inicial (pos0).
        tol : float
            Tolerancia en norma L2 (misma unidad que tus datos).
        ignore_nan : bool
            Si True, ignora NaNs al calcular el máximo en el tiempo.
            Si un nodo es todo NaN en el tiempo, NO se considera estático.
        return_mask : bool
            Si True, devuelve (idx, mask_bool).

        Retorna
        -------
        idx : np.ndarray
            Índices de nodos estáticos.
        mask_bool : np.ndarray (opcional)
            Máscara booleana de tamaño N con True en nodos estáticos.
        """
        # a numpy
        if isinstance(disp, torch.Tensor):
            disp = disp.detach().cpu().numpy()
        disp = np.asarray(disp)
        if disp.ndim != 3 or disp.shape[-1] != 3:
            raise ValueError("disp debe tener shape (T, N, 3)")

        # norma L2^2 por nodo y timestep
        sq = np.sum(disp * disp, axis=2)  # (T, N)

        if ignore_nan:
            max_sq = np.nanmax(sq, axis=0)          # (N,)
            all_nan = np.all(np.isnan(sq), axis=0)  # nodos sin datos válidos
            static_mask = (max_sq <= tol * tol) & (~all_nan)
        else:
            max_sq = np.max(sq, axis=0)
            static_mask = (max_sq <= tol * tol)

        idx = np.nonzero(static_mask)[0]
        return (idx, static_mask) if return_mask else idx
    @staticmethod
    def _build_and_adjacency_edge_index_from_vtk(dataset, bidirectional=True, clique=True):
        """
        elements: lista de elementos, cada uno = [old_id_i, old_id_j, ...]
        bidirectional: duplica aristas (i,j) y (j,i)
        clique: si True conecta todos los pares del elemento (más denso y suele ir mejor).
                si False conecta solo pares consecutivos (anillo).
        """
        nodes_elements_index = {}

        undirected_pairs = set()
        for i in range(dataset.GetNumberOfCells()):
            cell = dataset.GetCell(i)  # vtkCell (ej. vtkQuad, vtkHexahedron…)
            ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]

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
    
    def __build_edge_attr(self, edge_index, pos0):
            src, dst = edge_index
            rel_vec = pos0[dst] - pos0[src]            # (E, 3)
            length = rel_vec.norm(dim=-1, keepdim=True) # (E, 1)
            direction = rel_vec / (length + 1e-9)       # (E, 3)
            edge_attr = torch.cat([length, direction], dim=-1)  # (E, 4)
            return edge_attr
    
    def __avg_element_field_to_node(self, nodes_elements_index, elem_field_1d, N=None):
        """
        nodes_elements_index: dict {node_id: set(elem_ids)}
        elem_field_1d: np.ndarray shape (E,)
        N: nº de nodos (opcional; si None, usa max key + 1)
        """
        if N is None:
            N = max(nodes_elements_index.keys()) + 1
        out = np.zeros((N,), dtype=np.float32)
        for n, elems in nodes_elements_index.items():
            if len(elems) == 0:
                out[n] = 0.0
            else:
                vals = elem_field_1d[list(elems)]
                out[n] = float(np.nanmean(vals))
        return out  
    

    def check_vel_disp_consistency(self, all_displacements, all_velocities, all_time_steps, name=""):
        """
        all_displacements: np.ndarray (T, N, 3)
        all_velocities:    np.ndarray (T, N, 3)
        all_time_steps:    list/np.ndarray (T,) con dt[0]=0 y dt[t]=time[t]-time[t-1]
        """
        disp = np.asarray(all_displacements, dtype=np.float64)
        vel  = np.asarray(all_velocities,    dtype=np.float64)
        dt    = np.asarray(all_time_steps,   dtype=np.float64)

        T, N, C = disp.shape
        assert vel.shape == disp.shape, f"vel shape {vel.shape} != disp shape {disp.shape}"
        assert dt.shape[0] == T, f"time_steps len {dt.shape[0]} != T {T}"
        assert C == 3, "Se esperaba 3 componentes (x,y,z)"

        # Diferencias entre t y t+1
        dx = disp[1:] - disp[:-1]                # (T-1, N, 3)
        dt_steps = dt[1:].reshape(-1, 1, 1)      # (T-1, 1, 1) para broadcasting

        v_fwd  = vel[:-1]                        # v_t
        v_back = vel[1:]                         # v_{t+1}
        v_trap = 0.5 * (v_fwd + v_back)          # (v_t + v_{t+1})/2

        pred_fwd  = v_fwd  * dt_steps
        pred_back = v_back * dt_steps
        pred_trap = v_trap * dt_steps

        err_fwd  = dx - pred_fwd
        err_back = dx - pred_back
        err_trap = dx - pred_trap

        # Métricas: RMSE absoluto y relativo (respecto a ||dx||)
        eps = 1e-12
        def rmse(a): return np.sqrt(np.mean(a**2))
        def rel_rmse(err, ref): return rmse(err) / (rmse(ref) + eps)

        rmse_dx      = rmse(dx)
        rmse_fwd     = rmse(err_fwd)
        rmse_back    = rmse(err_back)
        rmse_trap    = rmse(err_trap)

        rrel_fwd  = rel_rmse(err_fwd,  dx)
        rrel_back = rel_rmse(err_back, dx)
        rrel_trap = rel_rmse(err_trap, dx)

        # También por componente (x,y,z), por si quieres ver asimetrías
        comp = ['x','y','z']
        comp_stats = {}
        for i in range(3):
            comp_stats[comp[i]] = dict(
                rmse_dx   = rmse(dx[:,:,i]),
                rmse_fwd  = rmse(err_fwd[:,:,i]),
                rmse_back = rmse(err_back[:,:,i]),
                rmse_trap = rmse(err_trap[:,:,i]),
                rel_fwd   = rel_rmse(err_fwd[:,:,i],  dx[:,:,i]),
                rel_back  = rel_rmse(err_back[:,:,i], dx[:,:,i]),
                rel_trap  = rel_rmse(err_trap[:,:,i], dx[:,:,i]),
            )

        # Resumen
        print(f"\n=== Consistencia Δ vs v {('('+name+')') if name else ''} ===")
        print(f"T={T}, N={N}")
        print(f"RMSE(dx):      {rmse_dx: .6e}")
        print(f"RMSE forward:  {rmse_fwd: .6e}   (rel {rrel_fwd: .3%})")
        print(f"RMSE backward: {rmse_back: .6e}  (rel {rrel_back: .3%})")
        print(f"RMSE trapz:    {rmse_trap: .6e}  (rel {rrel_trap: .3%})")
        best = min([("forward", rrel_fwd), ("backward", rrel_back), ("trapezoidal", rrel_trap)], key=lambda x: x[1])
        print(f"▶ Mejor esquema (global): {best[0]}")

        print("\nPor componente:")
        for ax in comp:
            s = comp_stats[ax]
            print(f"  {ax}: RMSE(dx)={s['rmse_dx']: .3e} | fwd={s['rmse_fwd']: .3e} ({s['rel_fwd']:.2%})"
                f"  back={s['rmse_back']: .3e} ({s['rel_back']:.2%})  trap={s['rmse_trap']: .3e} ({s['rel_trap']:.2%})")

        # Devuelve por si quieres usarlo programáticamente
        return {
            "global": {
                "rmse_dx": rmse_dx,
                "forward": {"rmse": rmse_fwd, "rel": rrel_fwd},
                "backward":{"rmse": rmse_back,"rel": rrel_back},
                "trapz":   {"rmse": rmse_trap,"rel": rrel_trap},
                "best": best[0],
            },
            "per_component": comp_stats
        }


if __name__ == "__main__":
    postprocess = D3PlotPostProcess()
    # input_file = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/simulation_results_copy/Geometry-017/d3plot"
    # input_file = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/d3plots/Geometry-0/d3plot"
    # postprocess.process(input_file, "0", time_step=5)
    input_dir = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/d3plots/"
    output_dir = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/d3plots/graphs"
    postprocess.process_all(input_dir, output_dir, time_step=5)