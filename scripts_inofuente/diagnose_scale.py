"""
Diagnostico de escala: compara las longitudes de segmento del humano (crudas y
escaladas por GMR) contra las del robot.

Motivacion: con la config derivada, las rodillas del G1 saturan en su limite de
EXTENSION (-5 grados, pierna recta) el ~59% de los frames. Que saturen al
estirarse, y no al flexionarse, apunta a que los objetivos de pie quedan
demasiado lejos de la cadera: el robot estira la pierna al maximo y aun no llega.
Eso es un problema de escala, no de pesos del IK.

La escala depende de actual_human_height, que en twh_loader.py se ESTIMA como
altura del hueso de la cabeza + un margen de coronilla fijado a mano. Este script
comprueba si esa estimacion es razonable midiendo las proporciones reales.

Uso:
    python scripts_inofuente/diagnose_scale.py
    python scripts_inofuente/diagnose_scale.py --human_height 1.70
"""

import argparse
import contextlib
import io
import json
import pathlib
import sys

import mujoco as mj
import numpy as np
from rich import print

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.params import ROBOT_XML_DICT

from twh_loader import load_twh_bvh_file
from twh_to_robot import register_twh_configs

DERIVED = HERE / "ik_configs" / "bvh_twh_to_g1_derived.json"
DEFAULT_TWH = (
    HERE.parent / "motion_data" / "twh_genea_challenge" / "genea2023_trn"
    / "genea2023_dataset" / "trn" / "main-agent" / "bvh"
    / "trn_2023_v0_000_main-agent.bvh"
)

# Cadena de la pierna izquierda: (link del robot, hueso humano)
CADENA_PIERNA = [
    ("pelvis", "b_root"),
    ("left_hip_yaw_link", "b_l_upleg"),
    ("left_knee_link", "b_l_leg"),
    ("left_ankle_roll_link", "b_l_foot_mod"),
]
CADENA_BRAZO = [
    ("torso_link", "b_spine3"),
    ("left_shoulder_yaw_link", "b_l_arm"),
    ("left_elbow_link", "b_l_forearm"),
    ("left_wrist_yaw_link", "b_l_wrist"),
]


def robot_segment_lengths(robot, cadena):
    """Longitudes de segmento del robot en su pose por defecto (son fijas)."""
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    pos = {}
    for link, _ in cadena:
        pos[link] = data.xpos[model.body(link).id].copy()
    longs = []
    for i in range(len(cadena) - 1):
        a, b = cadena[i][0], cadena[i + 1][0]
        longs.append((f"{a} -> {b}", float(np.linalg.norm(pos[b] - pos[a]))))
    return longs


def human_segment_lengths(frame, cadena):
    longs = []
    for i in range(len(cadena) - 1):
        a, b = cadena[i][1], cadena[i + 1][1]
        pa = np.asarray(frame[a][0])
        pb = np.asarray(frame[b][0])
        longs.append((f"{a} -> {b}", float(np.linalg.norm(pb - pa))))
    return longs


def report(nombre, robot_longs, crudo_longs, escalado_longs):
    print(f"\n[bold cyan]{nombre}[/bold cyan]")
    print(f"  {'segmento':44s} {'robot':>8s} {'humano':>8s} {'escalado':>9s} {'ratio':>7s}")
    tot_r = tot_e = 0.0
    for (nr, lr), (_, lc), (ne, le) in zip(robot_longs, crudo_longs, escalado_longs):
        ratio = le / lr if lr > 1e-6 else float("nan")
        color = "green" if 0.85 <= ratio <= 1.15 else ("yellow" if 0.7 <= ratio <= 1.3 else "red")
        etiqueta = ne.split(" -> ")[0][:42]
        print(f"  {etiqueta:44s} {lr:8.3f} {lc:8.3f} {le:9.3f} "
              f"[{color}]{ratio:7.2f}[/{color}]")
        tot_r += lr
        tot_e += le
    ratio_tot = tot_e / tot_r if tot_r > 1e-6 else float("nan")
    color = "green" if 0.9 <= ratio_tot <= 1.1 else "red"
    print(f"  {'TOTAL de la cadena':44s} {tot_r:8.3f} {'':8s} {tot_e:9.3f} "
          f"[{color}]{ratio_tot:7.2f}[/{color}]")
    return ratio_tot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twh_bvh", default=str(DEFAULT_TWH))
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--human_height", type=float, default=None,
                        help="Fuerza la estatura en lugar de estimarla.")
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()

    register_twh_configs(ik_config=DERIVED, robot=args.robot)
    frames, altura = load_twh_bvh_file(args.twh_bvh, human_height=args.human_height)

    with open(DERIVED) as f:
        cfg = json.load(f)
    supuesta = cfg["human_height_assumption"]
    ratio_escala = altura / supuesta

    print(f"Estatura usada        : {altura:.3f} m")
    print(f"human_height_assumption: {supuesta:.3f} m  (heredado de la config LAFAN1)")
    print(f"Ratio de escala        : {ratio_escala:.4f}")

    # Proporciones del sujeto, para juzgar si la estatura estimada es plausible
    f0 = frames[args.frame]
    pelvis_z = float(np.asarray(f0["b_root"][0])[2])
    cabeza_z = float(np.asarray(f0["b_head"][0])[2])
    print(f"\nProporciones del sujeto (frame {args.frame}):")
    print(f"  pelvis a {pelvis_z:.3f} m = {100 * pelvis_z / altura:.1f}% de la estatura "
          f"(tipico en adultos: 52-55%)")
    print(f"  hueso de cabeza a {cabeza_z:.3f} m = {100 * cabeza_z / altura:.1f}% "
          f"(tipico ojos/craneo: 90-94%)")

    # Datos escalados tal y como los ve el IK
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        retargeter = GMR(src_human="bvh_twh", tgt_robot=args.robot,
                         actual_human_height=altura, verbose=False)
        retargeter.retarget(frames[args.frame])
    escalado = retargeter.scaled_human_data

    for nombre, cadena in (("PIERNA IZQUIERDA", CADENA_PIERNA), ("BRAZO IZQUIERDO", CADENA_BRAZO)):
        report(
            nombre,
            robot_segment_lengths(args.robot, cadena),
            human_segment_lengths(f0, cadena),
            human_segment_lengths(escalado, cadena),
        )

    print("\n[yellow]AVISO sobre los ratios de arriba[/yellow]: los links del robot no "
          "coinciden con los\n  landmarks anatomicos del humano. left_hip_yaw_link es el "
          "TERCER link de la cadena de\n  cadera (pitch->roll->yaw), asi que "
          "'pelvis -> hip_yaw' ya incluye parte del muslo.\n  Los ratios por segmento "
          "mezclan diferencia de escala con diferencia de colocacion de\n  marcos, y no "
          "son concluyentes por si solos. Lo que decide es el ALCANCE:")

    # Alcance: distancia directa pelvis->tobillo frente a lo que el robot puede
    # cubrir con la pierna recta. Es la restriccion que realmente satura la rodilla.
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    p_pelvis = data.xpos[model.body("pelvis").id]
    p_ankle = data.xpos[model.body("left_ankle_roll_link").id]
    alcance_robot = float(np.linalg.norm(p_ankle - p_pelvis))

    def span(frame, a, b):
        return float(np.linalg.norm(np.asarray(frame[b][0]) - np.asarray(frame[a][0])))

    print(f"\n[bold cyan]ALCANCE pelvis -> tobillo[/bold cyan]")
    print(f"  robot en pose por defecto        : {alcance_robot:.3f} m")

    d_crudo = span(f0, "b_root", "b_l_foot_mod")
    d_escalado = span(escalado, "b_root", "b_l_foot_mod")
    print(f"  humano TWH crudo                 : {d_crudo:.3f} m")
    print(f"  humano TWH escalado (lo que ve el IK): {d_escalado:.3f} m  "
          f"ratio {d_escalado / alcance_robot:.3f}")

    # Asimetria izquierda/derecha en los datos HUMANOS. Sirve para distinguir si una
    # asimetria observada en el robot viene del sujeto (que reparte el peso en una
    # pierna) o la introduce el retargeting.
    print(f"\n[bold cyan]ASIMETRIA IZQUIERDA/DERECHA en los datos humanos[/bold cyan]")
    for lado, etq in (("l", "izquierda"), ("r", "derecha")):
        angs, alturas_pie = [], []
        for fr in frames:
            cadera = np.asarray(fr[f"b_{lado}_upleg"][0])
            rodilla = np.asarray(fr[f"b_{lado}_leg"][0])
            tobillo = np.asarray(fr[f"b_{lado}_foot"][0])
            v1 = rodilla - cadera
            v2 = tobillo - rodilla
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            cos = np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)
            angs.append(np.degrees(np.arccos(cos)))  # 0 = pierna recta
            alturas_pie.append(tobillo[2])
        angs = np.array(angs)
        alturas_pie = np.array(alturas_pie)
        print(f"  pierna {etq:9s}: flexion de rodilla media {angs.mean():5.1f} deg "
              f"(min {angs.min():4.1f}, max {angs.max():5.1f})   "
              f"tobillo z medio {alturas_pie.mean():.3f} m")

    # Anchura de apoyo. El human_scale_table aplica una escala ISOTROPA a todos los
    # landmarks de la pierna, asi que al acortar el miembro tambien junta los pies.
    # Si el robot no puede separarlos, las piernas salen estrechas o cruzadas.
    p_ankle_l = data.xpos[model.body("left_ankle_roll_link").id]
    p_ankle_r = data.xpos[model.body("right_ankle_roll_link").id]
    ancho_robot = float(np.linalg.norm(p_ankle_l - p_ankle_r))

    def ancho(frame):
        a = np.asarray(frame["b_l_foot_mod"][0])
        b = np.asarray(frame["b_r_foot_mod"][0])
        return float(np.linalg.norm((a - b)[:2]))  # solo en el plano del suelo

    print(f"\n[bold cyan]ANCHURA DE APOYO (separacion entre tobillos)[/bold cyan]")
    print(f"  robot en pose por defecto        : {ancho_robot:.3f} m")
    a_crudo, a_esc = ancho(f0), ancho(escalado)
    print(f"  humano TWH crudo                 : {a_crudo:.3f} m")
    print(f"  humano TWH escalado              : {a_esc:.3f} m  "
          f"(la escala la reduce un {100 * (1 - a_esc / a_crudo):.0f}%)")
    if a_esc < ancho_robot:
        print(f"  [yellow]El objetivo pide los pies {1000 * (ancho_robot - a_esc):.0f} mm mas "
              f"juntos que la postura neutra del robot.[/yellow]")

    if d_escalado > alcance_robot:
        print(f"  [red]El objetivo excede el alcance en {1000 * (d_escalado - alcance_robot):.0f} mm: "
              f"la rodilla se estira hasta su limite y aun no llega.[/red]")
    else:
        margen = alcance_robot - d_escalado
        print(f"  [green]El objetivo esta dentro del alcance, con {1000 * margen:.0f} mm "
              f"de margen (la rodilla debe flexionar).[/green]")


if __name__ == "__main__":
    main()
