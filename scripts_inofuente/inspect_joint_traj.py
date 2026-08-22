"""
Imprime la trayectoria temporal de articulaciones concretas de un movimiento
retargeteado, en texto.

Motivacion: el IK de GMR es incremental (integra velocidades partiendo de la
solucion del frame anterior, motion_retarget.py:184). Eso lo hace vulnerable a
minimos locales: si una articulacion queda atrapada contra su limite, puede no
recuperarse en el resto de la secuencia. Un valor medio o un porcentaje de
saturacion no distingue "atrapada desde el principio" de "toca el limite de vez
en cuando", y la diferencia cambia por completo el diagnostico.

Uso:
    python scripts_inofuente/inspect_joint_traj.py --pkl <motion.pkl>
    python scripts_inofuente/inspect_joint_traj.py --pkl <m.pkl> --joints left_knee_joint,right_knee_joint
"""

import argparse
import pathlib
import pickle
import sys

import mujoco as mj
import numpy as np
from rich import print

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

from general_motion_retargeting.params import ROBOT_XML_DICT

DEFAULT_JOINTS = (
    "left_knee_joint,right_knee_joint,"
    "left_hip_pitch_joint,right_hip_pitch_joint,"
    "waist_pitch_joint"
)

BLOQUES = " .:-=+*#%@"  # rampa de densidad para la barra de texto


def barra(valor, lo, hi, ancho=40):
    """Posicion del valor dentro de su rango, como barra de texto."""
    if hi - lo < 1e-9:
        return " " * ancho
    t = (valor - lo) / (hi - lo)
    t = min(max(t, 0.0), 1.0)
    pos = int(round(t * (ancho - 1)))
    linea = ["-"] * ancho
    linea[pos] = "#"
    return "".join(linea)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--joints", default=DEFAULT_JOINTS)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        d = pickle.load(f)
    dof_pos = np.asarray(d["dof_pos"])
    n = len(dof_pos)

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))

    # dof_pos empieza tras los 7 valores del free joint de la raiz
    nombres = [j.strip() for j in args.joints.split(",")]
    idx = np.linspace(0, n - 1, args.samples, dtype=int)

    for nombre in nombres:
        try:
            jid = model.joint(nombre).id
        except KeyError:
            print(f"[red]{nombre}: no existe en el modelo[/red]")
            continue
        col = model.jnt_qposadr[jid] - 7
        if col < 0 or col >= dof_pos.shape[1]:
            print(f"[red]{nombre}: fuera de rango de dof_pos[/red]")
            continue

        lo, hi = np.rad2deg(model.jnt_range[jid])
        vals = np.rad2deg(dof_pos[:, col])
        recorrido = vals.max() - vals.min()
        pct_lo = 100.0 * np.mean(vals <= lo + 2.0)

        color = "red" if recorrido < 15 else ("yellow" if recorrido < 30 else "green")
        print(f"\n[bold]{nombre}[/bold]  limite [{lo:.0f}, {hi:.0f}] deg   "
              f"recorrido usado [{color}]{recorrido:.1f} deg[/{color}]   "
              f"en el limite inferior el {pct_lo:.0f}% de frames")
        for i in idx:
            print(f"    f{i:5d}  {vals[i]:7.1f} deg  |{barra(vals[i], lo, hi)}|")


if __name__ == "__main__":
    main()
