"""
Diagnostico de convencion de ejes de hueso entre dos esqueletos BVH.

Motivacion: al clonar bvh_lafan1_to_g1.json cambiando solo los nombres de huesos,
el retargeting de TWH sale muy degradado (residuo IK ~5x peor, 17 DoF saturados).
La hipotesis es que el problema NO son los pesos, sino los cuaterniones rot_offset
de la config: codifican la transformada del marco del hueso humano al marco del
link del robot, y solo son validos para la convencion de ejes de LAFAN1.

Este script mide, para cada hueso, en que eje de su PROPIO marco local corre el
hueso (direccion hacia su hijo). Si LAFAN1 y TWH usan ejes distintos, los
rot_offset heredados son sistematicamente incorrectos y hay que recalcularlos.

Uso:
    python scripts_inofuente/diagnose_bone_axes.py
"""

import pathlib

import numpy as np
from rich import print
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh

GMR_ROOT = pathlib.Path(__file__).parent.parent

LAFAN1_BVH = GMR_ROOT / "motion_data" / "lafan1" / "dance1_subject1.bvh"
TWH_BVH = (
    GMR_ROOT / "motion_data" / "twh_genea_challenge" / "genea2023_trn"
    / "genea2023_dataset" / "trn" / "main-agent" / "bvh"
    / "trn_2023_v0_000_main-agent.bvh"
)

# Pares (padre, hijo) equivalentes en ambos esqueletos, en el mismo orden
PAIRS_LAFAN1 = [
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
    ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFoot"),
    ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFoot"),
    ("Hips", "Spine"),
    ("Spine2", "Neck"),
]
PAIRS_TWH = [
    ("b_l_arm", "b_l_forearm"),
    ("b_l_forearm", "b_l_wrist"),
    ("b_r_arm", "b_r_forearm"),
    ("b_r_forearm", "b_r_wrist"),
    ("b_l_upleg", "b_l_leg"),
    ("b_l_leg", "b_l_foot"),
    ("b_r_upleg", "b_r_leg"),
    ("b_r_leg", "b_r_foot"),
    ("b_root", "b_spine0"),
    ("b_spine3", "b_neck0"),
]


def bone_axes(bvh_file, pairs, label, frame=0):
    data = read_bvh(str(bvh_file))
    idx = {b: i for i, b in enumerate(data.bones)}
    quats, pos = utils.quat_fk(data.quats, data.pos, data.parents)

    print(f"\n[bold cyan]{label}[/bold cyan]  ({pathlib.Path(bvh_file).name})")
    print("  direccion hacia el hijo, expresada en el marco LOCAL del hueso padre:")

    resultado = {}
    for parent, child in pairs:
        if parent not in idx or child not in idx:
            print(f"    {parent:14s} -> {child:14s}: [red]no encontrado[/red]")
            continue

        q_parent = quats[frame, idx[parent]]
        v = pos[frame, idx[child]] - pos[frame, idx[parent]]
        norma = np.linalg.norm(v)
        if norma < 1e-6:
            print(f"    {parent:14s} -> {child:14s}: [yellow]longitud nula[/yellow]")
            continue

        local = R.from_quat(q_parent, scalar_first=True).inv().apply(v / norma)
        k = int(np.argmax(np.abs(local)))
        eje = ("+" if local[k] > 0 else "-") + "XYZ"[k]
        resultado[(parent, child)] = eje
        print(f"    {parent:14s} -> {child:14s}: "
              f"[{local[0]:6.3f} {local[1]:6.3f} {local[2]:6.3f}]  eje = [bold]{eje}[/bold]")

    return resultado


def main():
    ejes_laf = bone_axes(LAFAN1_BVH, PAIRS_LAFAN1, "LAFAN1")
    ejes_twh = bone_axes(TWH_BVH, PAIRS_TWH, "TWH / GENEA 2023")

    print(f"\n[bold yellow]{'=' * 66}[/bold yellow]")
    print("[bold yellow]Comparativa de convencion por hueso equivalente[/bold yellow]")
    print(f"[bold yellow]{'=' * 66}[/bold yellow]")

    coinciden = 0
    total = 0
    for (pl, cl), (pt, ct) in zip(PAIRS_LAFAN1, PAIRS_TWH):
        a = ejes_laf.get((pl, cl))
        b = ejes_twh.get((pt, ct))
        if a is None or b is None:
            continue
        total += 1
        igual = a == b
        coinciden += igual
        marca = "[green]coincide[/green]" if igual else "[red]DISTINTO[/red]"
        print(f"  {pl:14s} ({a})  vs  {pt:14s} ({b})   {marca}")

    print(f"\n  Coincidencias: {coinciden}/{total}")
    if coinciden < total:
        print("  [red]Los rot_offset heredados de la config de LAFAN1 no son validos.[/red]")
        print("  [red]Hay que recalcularlos para la convencion de ejes de TWH.[/red]")
    else:
        print("  [green]Misma convencion: los rot_offset heredados deberian servir.[/green]")


if __name__ == "__main__":
    main()
