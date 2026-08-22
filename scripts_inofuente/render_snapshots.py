"""
Renderiza capturas del robot retargeteado a PNG, para inspeccion visual.

Genera una hoja de contactos: una fila por vista de camara y una columna por
frame, de modo que una sola imagen basta para juzgar postura y orientacion.

Complementa validate_retarget.py: los residuos numericos dicen CUANTO se
desvia, las capturas dicen EN QUE DIRECCION, que es lo que hace falta para
corregir marcos de referencia.

Uso:
    python scripts_inofuente/render_snapshots.py --pkl <motion.pkl> --out out.png
    python scripts_inofuente/render_snapshots.py --pkl <m.pkl> --frames 0,40,80
    python scripts_inofuente/render_snapshots.py --home --out home.png
"""

import argparse
import pathlib
import pickle

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from rich import print

from general_motion_retargeting.params import ROBOT_BASE_DICT, ROBOT_XML_DICT

# (nombre, azimut, elevacion). Dos azimuts ortogonales bastan para detectar
# orientaciones invertidas o giradas 90 grados.
# OJO: la vista que resulta "frontal" depende de hacia donde mire el sujeto en
# coordenadas de mundo, asi que las etiquetas son por azimut y no por anatomia.
VIEWS = [
    ("az=180", 180.0, -10.0),
    ("az=90", 90.0, -10.0),
]


def load_qpos_from_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    root_pos = np.asarray(d["root_pos"])
    root_rot_xyzw = np.asarray(d["root_rot"])
    dof_pos = np.asarray(d["dof_pos"])
    # el pkl guarda xyzw; mujoco qpos espera wxyz
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
    qpos = np.concatenate([root_pos, root_rot_wxyz, dof_pos], axis=1)
    return qpos, d.get("fps", 30)


def render_frames(robot, qpos_list, frames, width, height, show_frame, add_floor):
    xml_path = str(ROBOT_XML_DICT[robot])
    base_body = ROBOT_BASE_DICT[robot]

    model = None
    if add_floor:
        # Anade un plano de suelo y una luz para tener referencia de horizontal.
        # Se usa MjSpec en vez de manipular el XML como texto porque el modelo
        # referencia sus mallas por ruta relativa: from_xml_string perderia el
        # directorio base y fallaria al abrir los .STL.
        try:
            spec = mj.MjSpec.from_file(xml_path)
            geom = spec.worldbody.add_geom()
            geom.name = "__floor_snap"
            geom.type = mj.mjtGeom.mjGEOM_PLANE
            geom.size = [5.0, 5.0, 0.05]
            geom.rgba = [0.55, 0.55, 0.6, 1.0]
            light = spec.worldbody.add_light()
            light.pos = [0.0, 0.0, 3.0]
            light.dir = [0.0, 0.0, -1.0]
            model = spec.compile()
        except Exception as e:  # noqa: BLE001
            print(f"[yellow]No se pudo anadir el suelo ({e}); se renderiza sin el.[/yellow]")

    if model is None:
        model = mj.MjModel.from_xml_path(xml_path)

    data = mj.MjData(model)
    renderer = mj.Renderer(model, height=height, width=width)

    scene_option = mj.MjvOption()
    mj.mjv_defaultOption(scene_option)
    if show_frame:
        scene_option.frame = mj.mjtFrame.mjFRAME_WORLD

    base_id = model.body(base_body).id
    grid = []

    for view_name, azimuth, elevation in VIEWS:
        fila = []
        for fi in frames:
            data.qpos[: len(qpos_list[fi])] = qpos_list[fi]
            mj.mj_forward(model, data)

            cam = mj.MjvCamera()
            mj.mjv_defaultCamera(cam)
            cam.lookat = data.xpos[base_id].copy()
            cam.distance = 2.6
            cam.azimuth = azimuth
            cam.elevation = elevation

            renderer.update_scene(data, camera=cam, scene_option=scene_option)
            fila.append(renderer.render())
        grid.append((view_name, fila))

    return grid


def compose(grid, frames, out_path):
    """Apila las vistas en filas y los frames en columnas, con separadores."""
    sep = 4
    filas_img = []
    for _, fila in grid:
        h, w, _ = fila[0].shape
        franja = np.full((h, sep, 3), 255, dtype=np.uint8)
        piezas = []
        for j, img in enumerate(fila):
            piezas.append(img)
            if j < len(fila) - 1:
                piezas.append(franja)
        filas_img.append(np.concatenate(piezas, axis=1))

    ancho = filas_img[0].shape[1]
    franja_h = np.full((sep, ancho, 3), 255, dtype=np.uint8)
    piezas = []
    for i, f in enumerate(filas_img):
        piezas.append(f)
        if i < len(filas_img) - 1:
            piezas.append(franja_h)
    hoja = np.concatenate(piezas, axis=0)

    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path, hoja)

    vistas = ", ".join(v for v, _ in grid)
    print(f"[green]Escrito {out_path}[/green]")
    print(f"  filas (arriba a abajo): {vistas}")
    print(f"  columnas (izq a der)  : frames {list(frames)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", default=None, help="Movimiento retargeteado (.pkl).")
    parser.add_argument("--home", action="store_true",
                        help="Renderiza el robot en qpos por defecto, como referencia.")
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--frames", default=None,
                        help="Indices separados por comas. Por defecto, 5 repartidos.")
    parser.add_argument("--width", type=int, default=420)
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--out", default="snapshots/out.png")
    parser.add_argument("--show_frame", action="store_true", default=True,
                        help="Dibuja los ejes del mundo como referencia.")
    parser.add_argument("--no_floor", action="store_true", default=False,
                        help="No inyectar plano de suelo.")
    args = parser.parse_args()

    if args.home:
        model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
        qpos_list = np.zeros((1, model.nq))
        qpos_list[0, 2] = 0.8  # altura razonable de pelvis
        qpos_list[0, 3] = 1.0  # cuaternion identidad wxyz
        frames = [0]
    elif args.pkl:
        qpos_list, fps = load_qpos_from_pkl(args.pkl)
        n = len(qpos_list)
        if args.frames:
            frames = [int(x) for x in args.frames.split(",")]
            frames = [min(max(f, 0), n - 1) for f in frames]
        else:
            frames = list(np.linspace(0, n - 1, 5, dtype=int))
        print(f"{n} frames a {fps} fps; capturando {frames}")
    else:
        parser.error("Indica --pkl o --home")

    grid = render_frames(
        args.robot, qpos_list, frames, args.width, args.height,
        args.show_frame, not args.no_floor,
    )
    compose(grid, frames, args.out)


if __name__ == "__main__":
    main()
