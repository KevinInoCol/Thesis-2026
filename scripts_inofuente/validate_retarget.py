"""
Validacion numerica del retargeting, para decidir con datos (y no a ojo) si una
config IK es aceptable o necesita tuning.

Ejecuta el retargeting sin visor sobre un BVH y reporta:
  - Residuo del IK por frame (error1/error2 de GMR, norma agregada de las tareas).
  - Residuo por tarea, separando posicion (m) y orientacion (rad), para saber
    QUE parte del cuerpo se esta ajustando mal.
  - Saturacion de articulaciones: DoF pegados a sus limites, sintoma tipico de
    una config mal escalada.

Permite comparar dos fuentes en la misma ejecucion, p.ej. TWH contra la linea
base LAFAN1, que es la comparacion que justifica cualquier tuning posterior.

Uso:
    python scripts_inofuente/validate_retarget.py --twh <fichero_twh.bvh>
    python scripts_inofuente/validate_retarget.py --twh <f.bvh> --lafan1 <f.bvh>
    python scripts_inofuente/validate_retarget.py --twh <f.bvh> --frames 500
"""

import argparse
import pathlib
import sys

import mujoco as mj
import numpy as np
from rich import print

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_bvh_file

from twh_loader import load_twh_bvh_file
from twh_to_robot import register_twh_configs

SAT_TOL_RAD = np.deg2rad(2.0)  # margen para considerar un DoF "pegado al limite"


def run(src_human, frames, robot, human_height, label):
    retargeter = GMR(
        src_human=src_human,
        tgt_robot=robot,
        actual_human_height=human_height,
        verbose=False,
    )

    n = len(frames)
    err1 = np.zeros(n)
    err2 = np.zeros(n)
    qpos_all = []

    # Residuos por tarea: mink devuelve un vector por tarea; para FrameTask son
    # 6 componentes (3 de posicion + 3 de rotacion).
    per_task_pos = {}
    per_task_rot = {}

    for i in range(n):
        qpos = retargeter.retarget(frames[i])
        qpos_all.append(qpos.copy())
        err1[i] = retargeter.error1()
        err2[i] = retargeter.error2()

        for table_idx, tasks in ((1, retargeter.tasks1), (2, retargeter.tasks2)):
            for task in tasks:
                e = task.compute_error(retargeter.configuration)
                name = f"t{table_idx}:{getattr(task, 'frame_name', type(task).__name__)}"
                if len(e) >= 6:
                    per_task_pos.setdefault(name, []).append(np.linalg.norm(e[:3]))
                    per_task_rot.setdefault(name, []).append(np.linalg.norm(e[3:6]))
                else:
                    per_task_pos.setdefault(name, []).append(np.linalg.norm(e))

    qpos_all = np.array(qpos_all)

    # Saturacion de articulaciones frente a los limites del modelo MuJoCo
    model = retargeter.model

    # Anchura de apoyo lograda, por cinematica directa sobre cada frame
    data_fk = mj.MjData(model)
    id_l = model.body("left_ankle_roll_link").id
    id_r = model.body("right_ankle_roll_link").id
    ancho_apoyo = np.zeros(len(qpos_all))
    for i, q in enumerate(qpos_all):
        data_fk.qpos[: len(q)] = q
        mj.mj_forward(model, data_fk)
        d = data_fk.xpos[id_l] - data_fk.xpos[id_r]
        ancho_apoyo[i] = np.linalg.norm(d[:2])
    sat = []
    for j in range(model.njnt):
        if model.jnt_type[j] != mj.mjtJoint.mjJNT_HINGE:
            continue
        if not model.jnt_limited[j]:
            continue
        qadr = model.jnt_qposadr[j]
        lo, hi = model.jnt_range[j]
        vals = qpos_all[:, qadr]
        n_lo = int(np.sum(vals <= lo + SAT_TOL_RAD))
        n_hi = int(np.sum(vals >= hi - SAT_TOL_RAD))
        if n_lo + n_hi > 0:
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j)
            sat.append({
                "joint": name,
                "pct_frames_saturado": 100.0 * (n_lo + n_hi) / len(vals),
                "en_min": n_lo,
                "en_max": n_hi,
                "rango_usado_deg": [float(np.rad2deg(vals.min())), float(np.rad2deg(vals.max()))],
                "limite_deg": [float(np.rad2deg(lo)), float(np.rad2deg(hi))],
            })

    return {
        "label": label,
        "frames": n,
        "err1": err1,
        "err2": err2,
        "qpos": qpos_all,
        "per_task_pos": {k: np.array(v) for k, v in per_task_pos.items()},
        "per_task_rot": {k: np.array(v) for k, v in per_task_rot.items()},
        "saturacion": sorted(sat, key=lambda d: -d["pct_frames_saturado"]),
        "ancho_apoyo": ancho_apoyo,
    }


def report(res):
    print(f"\n[bold cyan]{'=' * 72}[/bold cyan]")
    print(f"[bold cyan]{res['label']}  ({res['frames']} frames)[/bold cyan]")
    print(f"[bold cyan]{'=' * 72}[/bold cyan]")

    for k in ("err1", "err2"):
        e = res[k]
        print(f"  residuo IK {k}: media={e.mean():.4f}  p50={np.median(e):.4f}  "
              f"p95={np.percentile(e, 95):.4f}  max={e.max():.4f}")

    q = res["qpos"]
    print(f"  altura de la pelvis: min={q[:, 2].min():.3f} media={q[:, 2].mean():.3f} "
          f"max={q[:, 2].max():.3f} m")

    # Anchura de apoyo LOGRADA. Se mide en el resultado y no en el objetivo, porque
    # una postura que "parece" estrecha en una captura puede ser efecto de la
    # perspectiva de camara; el numero no engana.
    aw = res.get("ancho_apoyo")
    if aw is not None:
        cruces = 100.0 * np.mean(aw < 0.05)
        print(f"  anchura de apoyo   : min={aw.min():.3f} media={aw.mean():.3f} "
              f"max={aw.max():.3f} m   (por debajo de 50 mm el {cruces:.0f}% de frames)")

    print("\n  [bold]Residuo por tarea (peor primero, media sobre frames):[/bold]")
    filas = []
    for name, pos in res["per_task_pos"].items():
        rot = res["per_task_rot"].get(name)
        filas.append((name, pos.mean(), rot.mean() if rot is not None else float("nan")))
    for name, p, r in sorted(filas, key=lambda t: -t[1])[:16]:
        rtxt = f"{np.rad2deg(r):6.2f} deg" if not np.isnan(r) else "     n/a"
        print(f"    {name:34s} pos={p:.4f} m   rot={rtxt}")

    print("\n  [bold]Articulaciones saturadas:[/bold]")
    if not res["saturacion"]:
        print("    [green]ninguna[/green]")
    else:
        for s in res["saturacion"][:12]:
            color = "red" if s["pct_frames_saturado"] > 50 else "yellow"
            print(f"    [{color}]{s['joint']:28s} {s['pct_frames_saturado']:5.1f}% de frames "
                  f"(usado {s['rango_usado_deg'][0]:7.1f}..{s['rango_usado_deg'][1]:7.1f} deg, "
                  f"limite {s['limite_deg'][0]:7.1f}..{s['limite_deg'][1]:7.1f})[/{color}]")


def compare(a, b):
    print(f"\n[bold yellow]{'=' * 72}[/bold yellow]")
    print(f"[bold yellow]{a['label']}  vs  {b['label']}[/bold yellow]")
    print(f"[bold yellow]{'=' * 72}[/bold yellow]")
    for k in ("err1", "err2"):
        print(f"  residuo {k} medio : {a[k].mean():.4f}  vs  {b[k].mean():.4f}"
              f"   (ratio {a[k].mean() / b[k].mean():.2f}x)")
    print(f"  DoF saturados     : {len(a['saturacion'])}  vs  {len(b['saturacion'])}")
    peor_a = a["saturacion"][0]["pct_frames_saturado"] if a["saturacion"] else 0.0
    peor_b = b["saturacion"][0]["pct_frames_saturado"] if b["saturacion"] else 0.0
    print(f"  peor saturacion   : {peor_a:.1f}%  vs  {peor_b:.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twh", required=True, help="BVH de TWH/GENEA.")
    parser.add_argument("--lafan1", default=None, help="BVH de LAFAN1 como linea base.")
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--frames", type=int, default=300, help="Frames a evaluar por fuente.")
    parser.add_argument("--ik_config", default=None,
                        help="Config IK alternativa para TWH (p.ej. la derivada).")
    parser.add_argument("--human_height", type=float, default=None,
                        help="Fuerza la estatura del sujeto en lugar de estimarla.")
    args = parser.parse_args()

    register_twh_configs(ik_config=args.ik_config, robot=args.robot)

    twh_frames, twh_h = load_twh_bvh_file(
        args.twh, human_height=args.human_height, verbose=True
    )
    res_twh = run("bvh_twh", twh_frames[: args.frames], args.robot, twh_h,
                  f"TWH  {pathlib.Path(args.twh).name}")
    report(res_twh)

    if args.lafan1:
        laf_frames, laf_h = load_bvh_file(args.lafan1, format="lafan1")
        res_laf = run("bvh_lafan1", laf_frames[: args.frames], args.robot, laf_h,
                      f"LAFAN1  {pathlib.Path(args.lafan1).name}")
        report(res_laf)
        compare(res_twh, res_laf)


if __name__ == "__main__":
    main()
