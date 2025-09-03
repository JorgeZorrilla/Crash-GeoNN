import os
import numpy as np
import torch
import vtk
from vtk.util import numpy_support


def write_vtk_time_series_from_pyg(
    database,
    out_dir: str,
    deltas_are_wrt_pos0: bool = True,
    add_rest_frame: bool = False,
    timestep: float | None = None,
    basename: str = "frame",
    # --- NUEVO: opciones de animación / video ---
    make_animation: bool = True,              # activa/desactiva renderizado/exports
    video_path: str | None = None,            # si None, intenta out_dir/animacion.mp4
    export_pngs: bool = False,                # guarda también frames como PNG (secuencias)
    png_dir: str | None = None,               # si None, usa out_dir/frames_png
    fps: int = 30,
    size: tuple[int, int] = (800, 800),
    bg_color: tuple[float, float, float] = (0.05, 0.08, 0.12),
    line_width: float = 1.5,
    rotate_camera_deg_per_frame: float = 1.5, # animación simple: giro de cámara por frame
    offscreen: bool = True,                   # útil en servidores/headless
):
    """
    Crea una serie temporal VTK: frame_0000.vtp, frame_0001.vtp, ... + series.pvd
    y opcionalmente renderiza la animación y la exporta a video y/o PNGs.

    - Puntos: nodos
    - Líneas: a partir de edge_index (VTK_LINE)
    """

    os.makedirs(out_dir, exist_ok=True)

    # --- datos base ---
    pos0 = database[0].pos0        # (N,3) tensor
    edge_index = database[0].edge_index  # (2,E) tensor
    bc_mask = database[0].bc_mask
    rigid_mask = database[0].rigid_mask

    pos0_np = pos0.detach().cpu().numpy().astype(np.float64)
    N = pos0_np.shape[0]

    # Deltas por frame (T,N,3)
    deltas = torch.stack([d.x[:, :3] for d in database], dim=0).detach().cpu().numpy()

    # Construye secuencia de posiciones
    if deltas_are_wrt_pos0:
        pos_seq = pos0_np[None, ...] + deltas            # (T,N,3)
        if add_rest_frame:
            pos_seq = np.concatenate([pos0_np[None, ...], pos_seq], axis=0)  # (T+1,N,3)
    else:
        # deltas incrementales entre frames
        cum = deltas.cumsum(axis=0)
        pos_seq = pos0_np[None, ...] + cum               # (T,N,3)
        if add_rest_frame:
            pos_seq = np.concatenate([pos0_np[None, ...], pos_seq], axis=0)

    n_frames = pos_seq.shape[0]

    # --- construir conectividad VTK_LINE a partir de edge_index ---
    ei = edge_index.detach().cpu().numpy().astype(np.int64)
    # quitar duplicados y self-loops
    edges = {tuple(sorted((int(u), int(v)))) for u, v in ei.T if int(u) != int(v)}
    edges = np.array(sorted(list(edges)), dtype=np.int64)
    E = edges.shape[0]

    # cell array: para cada línea, [2, n0, n1]
    lines = vtk.vtkCellArray()
    for n0, n1 in edges:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, int(n0))
        line.GetPointIds().SetId(1, int(n1))
        lines.InsertNextCell(line)

    # --- helper para escribir un .vtp por frame ---
    def write_vtp(points_xyz: np.ndarray, path: str, lines: vtk.vtkCellArray):
        # vtkPoints
        pts = vtk.vtkPoints()
        pts.SetData(numpy_support.numpy_to_vtk(points_xyz, deep=True))

        # PolyData con puntos + líneas
        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        poly.SetLines(lines)

        # Escribir
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(path)
        writer.SetInputData(poly)
        writer.SetDataModeToBinary()
        writer.Write()

    # --- escribir todos los frames .vtp ---
    vtp_files = []
    for t in range(n_frames):
        fname = f"{basename}_{t:04d}.vtp"
        fpath = os.path.join(out_dir, fname)
        write_vtp(pos_seq[t], fpath, lines)
        vtp_files.append(fname)

    # --- escribir el .pvd (colección temporal) ---
    # Si no nos dan timestep, usamos 0,1,2,...
    if timestep is None:
        times = [float(i) for i in range(n_frames)]
    else:
        times = [float(i) * float(timestep) for i in range(n_frames)]

    pvd_path = os.path.join(out_dir, "series.pvd")
    with open(pvd_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        for t, vtp in enumerate(vtp_files):
            f.write(
                f'    <DataSet timestep="{times[t]}" group="" part="0" file="{vtp}"/>\n'
            )
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')

    print(f"[VTK] Escrito {n_frames} frames en: {out_dir}")
    print(f"[VTK] Abre en ParaView: {pvd_path}")

    # -------------------------------------------------------------------------
    # NUEVO: Render y exportación a video/PNGs (opcional)
    # -------------------------------------------------------------------------
    if not make_animation:
        return

    # Construimos una sola tubería VTK y actualizamos solo las posiciones por frame
    # PolyData base (misma topología; cambian coords)
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(pos_seq[0], deep=True))

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lines)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(line_width)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(*bg_color)

    renwin = vtk.vtkRenderWindow()
    renwin.AddRenderer(renderer)
    renwin.SetSize(int(size[0]), int(size[1]))
    if offscreen and hasattr(renwin, "SetOffScreenRendering"):
        renwin.SetOffScreenRendering(1)

    # Ajuste de cámara a los bounds iniciales
    renwin.Render()
    renderer.ResetCamera()
    renwin.Render()

    # Capturador de imágenes de ventana
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(renwin)
    # w2i.SetScale(1)  # puedes subir a 2x si quieres más resolución
    w2i.ReadFrontBufferOff()  # más rápido en offscreen
    w2i.Update()

    # Preparar PNGs si se piden
    if export_pngs:
        if png_dir is None:
            png_dir = os.path.join(out_dir, "frames_png")
        os.makedirs(png_dir, exist_ok=True)
        png_writer = vtk.vtkPNGWriter()

    # Preparar video (si FFmpeg está disponible)
    ffmpeg_ok = False
    video_writer = None
    if video_path is None:
        video_path = os.path.join(out_dir, "animacion.mp4")

    try:
        video_writer = vtk.vtkFFMPEGWriter()
        video_writer.SetFileName(video_path)
        video_writer.SetRate(int(fps))
        video_writer.SetQuality(2)  # 0-2 (2 es alta)
        video_writer.SetInputConnection(w2i.GetOutputPort())
        video_writer.Start()
        ffmpeg_ok = True
        print(f"[VTK] Exportando video con vtkFFMPEGWriter -> {video_path}")
    except Exception as e:
        print(f"[VTK] vtkFFMPEGWriter no disponible o falló ({e}). Se omite video y se guardan PNGs si export_pngs=True.")

    # Bucle de animación
    for t in range(n_frames):
        # actualizar posiciones
        pts_vtk = numpy_support.numpy_to_vtk(pos_seq[t], deep=True)
        points.SetData(pts_vtk)
        points.Modified()
        poly.Modified()

        # giro de cámara simple (puedes cambiar a Track, Dolly, etc.)
        if rotate_camera_deg_per_frame != 0.0:
            renderer.GetActiveCamera().Azimuth(float(rotate_camera_deg_per_frame))

        renwin.Render()
        w2i.Modified()
        w2i.Update()

        # escribir a video si disponible
        if ffmpeg_ok and video_writer is not None:
            video_writer.Write()

        # escribir PNG si se pidió
        if export_pngs:
            png_path = os.path.join(png_dir, f"{basename}_{t:04d}.png")
            png_writer.SetFileName(png_path)
            png_writer.SetInputConnection(w2i.GetOutputPort())
            png_writer.Write()

    # cerrar video si se abrió
    if ffmpeg_ok and video_writer is not None:
        video_writer.End()
        print(f"[VTK] Video escrito: {video_path}")

    if export_pngs:
        print(f"[VTK] Secuencia PNG escrita en: {png_dir}")


if __name__ == "__main__":
    input_file = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/d3plots/graphs/graph_0.pt"
    db = torch.load(input_file, map_location="cpu", weights_only=False)

    output_path = "C:/Users/jorge/Documents/M2i/Crash-GeoNN/vtk/"
    write_vtk_time_series_from_pyg(
        database=db,                        # tu list[Data]
        out_dir=output_path,                # carpeta de salida
        deltas_are_wrt_pos0=True,           # según confirmaste
        add_rest_frame=False,               # pon True si quieres un frame en pos0
        timestep=1e-4,                      # opcional: paso temporal para ParaView
        basename="frame",
        # --- NUEVO: opciones de animación ---
        make_animation=True,
        video_path=None,        # si None -> out_dir/animacion.mp4
        export_pngs=False,      # pon True para tener también PNGs
        fps=30,
        size=(900, 900),
        bg_color=(0.1, 0.15, 0.2),
        line_width=2.0,
        rotate_camera_deg_per_frame=2.0,
        offscreen=True,
    )
