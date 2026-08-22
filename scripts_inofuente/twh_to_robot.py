"""
Retargeting de BVH de Talking With Hands / GENEA Challenge a robot humanoide.

Equivalente a scripts/bvh_to_robot.py del upstream, pero para el esqueleto TWH.
No modifica ningun fichero del upstream: registra la config IK propia en
IK_CONFIG_DICT en tiempo de ejecucion (motion_retarget.py:57 hace una simple
busqueda en dict, asi que basta con inyectar la entrada antes de instanciar GMR).

Uso:
    python scripts_inofuente/twh_to_robot.py --bvh_file <fichero.bvh>
    python scripts_inofuente/twh_to_robot.py --bvh_file <f.bvh> --save_path out.pkl \
        --record_video --video_path out.mp4
"""

import argparse
import os
import pathlib
import sys
import time

import numpy as np
from rich import print
from tqdm import tqdm

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting import params

from twh_loader import load_twh_bvh_file

# Config IK propia, registrada bajo la clave "bvh_twh" sin tocar el upstream
TWH_IK_CONFIGS = {
    "unitree_g1": HERE / "ik_configs" / "bvh_twh_to_g1.json",
}


def register_twh_configs(ik_config=None, robot="unitree_g1"):
    """Inyecta la fuente "bvh_twh" en el IK_CONFIG_DICT del upstream.

    Si se pasa ik_config, se usa esa ruta para el robot indicado en lugar de la
    config por defecto (util para comparar la linea base contra la calibrada).
    """
    configs = dict(TWH_IK_CONFIGS)
    if ik_config is not None:
        configs[robot] = pathlib.Path(ik_config)

    faltan = [str(p) for p in configs.values() if not p.exists()]
    if faltan:
        raise FileNotFoundError(
            f"No existe la config IK: {faltan}\n"
            f"Generala primero con: python scripts_inofuente/make_twh_ik_config.py"
        )
    params.IK_CONFIG_DICT["bvh_twh"] = configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh_file", required=True, help="BVH de TWH/GENEA a retargetear.")
    parser.add_argument("--robot", choices=sorted(TWH_IK_CONFIGS.keys()), default="unitree_g1")
    parser.add_argument("--save_path", default=None, help="Ruta del .pkl de salida.")
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--video_path", default="videos/twh_example.mp4")
    parser.add_argument("--rate_limit", action="store_true", default=False,
                        help="Reproduce a la velocidad real del movimiento humano.")
    parser.add_argument("--motion_fps", type=int, default=30,
                        help="TWH/GENEA 2023 esta a 30 fps.")
    parser.add_argument("--human_height", type=float, default=None,
                        help="Estatura del sujeto en m. Por defecto se estima del frame 0.")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Procesa solo los primeros N frames (util para pruebas).")
    parser.add_argument("--headless", action="store_true", default=False,
                        help="Sin ventana interactiva (requiere --record_video para ver algo).")
    parser.add_argument("--ik_config", default=None,
                        help="Config IK alternativa (p.ej. la calibrada).")
    args = parser.parse_args()

    register_twh_configs(ik_config=args.ik_config, robot=args.robot)

    frames, human_height = load_twh_bvh_file(
        args.bvh_file, human_height=args.human_height, verbose=True
    )
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    print(f"[twh] {len(frames)} frames a {args.motion_fps} fps, "
          f"altura humana = {human_height:.3f} m")

    retargeter = GMR(
        src_human="bvh_twh",
        tgt_robot=args.robot,
        actual_human_height=human_height,
        verbose=False,
    )

    viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=args.motion_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=args.video_path,
    )

    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    qpos_list = []

    fps_counter, fps_start = 0, time.time()
    pbar = tqdm(total=len(frames), desc="Retargeting TWH")

    try:
        for i in range(len(frames)):
            fps_counter += 1
            now = time.time()
            if now - fps_start >= 2.0:
                print(f"FPS de retargeting: {fps_counter / (now - fps_start):.2f}")
                fps_counter, fps_start = 0, time.time()
            pbar.update(1)

            qpos = retargeter.retarget(frames[i])
            qpos_list.append(qpos.copy())

            viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                rate_limit=args.rate_limit,
                follow_camera=False,
            )
    finally:
        pbar.close()
        if args.save_path and qpos_list:
            import pickle

            motion_data = {
                "fps": args.motion_fps,
                # wxyz -> xyzw, igual que hace el upstream al guardar
                "root_pos": np.array([q[:3] for q in qpos_list]),
                "root_rot": np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list]),
                "dof_pos": np.array([q[7:] for q in qpos_list]),
                "local_body_pos": None,
                "link_body_list": None,
            }
            with open(args.save_path, "wb") as f:
                pickle.dump(motion_data, f)
            print(f"[green]Guardado en {args.save_path}[/green] "
                  f"({len(qpos_list)} frames)")
        viewer.close()


if __name__ == "__main__":
    main()
