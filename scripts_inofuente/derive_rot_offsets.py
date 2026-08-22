"""
Deriva los rot_offset de la config IK de TWH a partir de una pose de referencia
erguida, y los ajusta al grupo octaedrico.

Por que este metodo y no minimizar el residuo del IK
-----------------------------------------------------
calibrate_rot_offsets.py buscaba los offsets minimizando el residuo del IK. Eso
resulto ser un objetivo EQUIVOCADO: el residuo mide si el robot ALCANZA los
objetivos, no si los objetivos son anatomicamente CORRECTOS. Bajo el residuo de
4.10 a 1.76 mientras la postura seguia rota y la saturacion clavada en 13 DoF.

Aqui se resuelve el algebra directamente. Para un hueso en correspondencia:

    R_robot = R_lafan1 * offset_lafan1        (la config del upstream funciona)
    R_robot = R_twh    * offset_twh           (lo que buscamos)
    =>  offset_twh = Delta * offset_lafan1,   Delta = R_twh^-1 * R_lafan1

Delta solo depende de la convencion de ejes de cada esqueleto, no de la pose. Se
mide poniendo ambos esqueletos en la MISMA pose fisica. No hace falta que las dos
poses coincidan al grado: como las convenciones difieren en multiplos de 90
grados, basta acercarse y AJUSTAR al elemento mas proximo de las 24 rotaciones
que permutan ejes con signo. El ajuste absorbe el error de correspondencia y
devuelve una rotacion exacta e interpretable.

El angulo de ajuste (cuanto habia que girar la delta cruda para caer en el grupo)
es la comprobacion de honestidad del metodo: si es pequeno, la correspondencia de
poses era buena; si es grande, el frame de referencia elegido no sirve.

Uso:
    python scripts_inofuente/derive_rot_offsets.py
    python scripts_inofuente/derive_rot_offsets.py --lafan1_bvh motion_data/lafan1/walk1_subject1.bvh
"""

import argparse
import copy
import itertools
import json
import pathlib
import sys

import numpy as np
from rich import print
from scipy.spatial.transform import Rotation as R

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

from general_motion_retargeting.utils.lafan1 import load_bvh_file

from twh_loader import load_twh_bvh_file

BASE_CONFIG = HERE / "ik_configs" / "bvh_twh_to_g1.json"
OUT_CONFIG = HERE / "ik_configs" / "bvh_twh_to_g1_derived.json"

DEFAULT_TWH = (
    HERE.parent / "motion_data" / "twh_genea_challenge" / "genea2023_trn"
    / "genea2023_dataset" / "trn" / "main-agent" / "bvh"
    / "trn_2023_v0_000_main-agent.bvh"
)
DEFAULT_LAFAN1 = HERE.parent / "motion_data" / "lafan1" / "walk1_subject1.bvh"

# link del robot -> (hueso LAFAN1, hueso TWH). Son los 14 del ik_match_table.
BONE_MAP = {
    "pelvis": ("Hips", "b_root"),
    "torso_link": ("Spine2", "b_spine3"),
    "left_hip_yaw_link": ("LeftUpLeg", "b_l_upleg"),
    "left_knee_link": ("LeftLeg", "b_l_leg"),
    "left_ankle_roll_link": ("LeftFootMod", "b_l_foot_mod"),
    "right_hip_yaw_link": ("RightUpLeg", "b_r_upleg"),
    "right_knee_link": ("RightLeg", "b_r_leg"),
    "right_ankle_roll_link": ("RightFootMod", "b_r_foot_mod"),
    "left_shoulder_yaw_link": ("LeftArm", "b_l_arm"),
    "left_elbow_link": ("LeftForeArm", "b_l_forearm"),
    "left_wrist_yaw_link": ("LeftHand", "b_l_wrist"),
    "right_shoulder_yaw_link": ("RightArm", "b_r_arm"),
    "right_elbow_link": ("RightForeArm", "b_r_forearm"),
    "right_wrist_yaw_link": ("RightHand", "b_r_wrist"),
}

# Cadenas que comparten convencion de ejes. Se separan por LADO porque TWH espeja
# el brazo derecho (b_r_arm corre en -X mientras b_l_arm corre en +X) aunque NO
# espeja las piernas (ambas en -Y). Medido con diagnose_bone_axes.py.
#
# Los tobillos quedan FUERA de la cadena de la pierna a proposito: *FootMod es un
# pseudo-hueso sintetico y su convencion difiere de forma legitima (en TWH hereda
# la orientacion del pie, en LAFAN1 toma la del toe, que es un hueso real).
CHAINS = {
    "pelvis": ["pelvis"],
    "torso": ["torso_link"],
    "pierna_i": ["left_hip_yaw_link", "left_knee_link"],
    "pierna_d": ["right_hip_yaw_link", "right_knee_link"],
    "tobillo_i": ["left_ankle_roll_link"],
    "tobillo_d": ["right_ankle_roll_link"],
    "brazo_i": ["left_shoulder_yaw_link", "left_elbow_link", "left_wrist_yaw_link"],
    "brazo_d": ["right_shoulder_yaw_link", "right_elbow_link", "right_wrist_yaw_link"],
}

# Nombres necesarios para puntuar la pose, por esqueleto
LANDMARKS_LAFAN1 = {
    "pelvis": "Hips", "cuello": "Neck",
    "cadera_i": "LeftUpLeg", "rodilla_i": "LeftLeg", "tobillo_i": "LeftFoot",
    "cadera_d": "RightUpLeg", "rodilla_d": "RightLeg", "tobillo_d": "RightFoot",
    "hombro_i": "LeftArm", "codo_i": "LeftForeArm", "mano_i": "LeftHand",
    "hombro_d": "RightArm", "codo_d": "RightForeArm", "mano_d": "RightHand",
}
LANDMARKS_TWH = {
    "pelvis": "b_root", "cuello": "b_neck0",
    "cadera_i": "b_l_upleg", "rodilla_i": "b_l_leg", "tobillo_i": "b_l_foot",
    "cadera_d": "b_r_upleg", "rodilla_d": "b_r_leg", "tobillo_d": "b_r_foot",
    "hombro_i": "b_l_arm", "codo_i": "b_l_forearm", "mano_i": "b_l_wrist",
    "hombro_d": "b_r_arm", "codo_d": "b_r_forearm", "mano_d": "b_r_wrist",
}

ARRIBA = np.array([0.0, 0.0, 1.0])
ABAJO = np.array([0.0, 0.0, -1.0])


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def pose_score(frame, lm):
    """Puntua cuanto se parece un frame a "de pie, erguido, brazos bajados".

    Cada componente esta en [-1, 1]. Se necesita esa pose porque es la unica que
    podemos suponer identica en los dos datasets sin alinearlos explicitamente.
    """
    P = {k: np.asarray(frame[v][0]) for k, v in lm.items()}

    tronco = np.dot(unit(P["cuello"] - P["pelvis"]), ARRIBA)

    piernas_rectas, piernas_verticales, brazos_bajados = [], [], []
    for lado in ("i", "d"):
        muslo = unit(P[f"rodilla_{lado}"] - P[f"cadera_{lado}"])
        tibia = unit(P[f"tobillo_{lado}"] - P[f"rodilla_{lado}"])
        piernas_rectas.append(np.dot(muslo, tibia))
        piernas_verticales.append(
            np.dot(unit(P[f"tobillo_{lado}"] - P[f"cadera_{lado}"]), ABAJO)
        )
        brazo = unit(P[f"codo_{lado}"] - P[f"hombro_{lado}"])
        antebrazo = unit(P[f"mano_{lado}"] - P[f"codo_{lado}"])
        brazos_bajados.append(0.5 * (np.dot(brazo, ABAJO) + np.dot(antebrazo, ABAJO)))

    comp = {
        "tronco_vertical": float(tronco),
        "piernas_rectas": float(np.mean(piernas_rectas)),
        "piernas_verticales": float(np.mean(piernas_verticales)),
        "brazos_bajados": float(np.mean(brazos_bajados)),
    }
    comp["total"] = float(np.mean(list(comp.values())))
    return comp


def find_reference_frame(frames, lm, step, label):
    mejor_i, mejor = None, None
    for i in range(0, len(frames), step):
        c = pose_score(frames[i], lm)
        if mejor is None or c["total"] > mejor["total"]:
            mejor_i, mejor = i, c

    print(f"\n[bold]{label}[/bold]: frame de referencia = {mejor_i} "
          f"(de {len(frames)}, muestreo cada {step})")
    for k, v in mejor.items():
        color = "green" if v > 0.8 else ("yellow" if v > 0.5 else "red")
        print(f"    {k:20s} [{color}]{v:+.3f}[/{color}]")
    return mejor_i, mejor


def octahedral_rotations():
    """Las 24 rotaciones propias que permutan ejes con signo (det = +1)."""
    rots = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = signs[i]
            if np.isclose(np.linalg.det(M), 1.0):
                rots.append(M)
    assert len(rots) == 24
    return rots


def snap(delta, rots):
    """Ajusta delta a la rotacion del grupo mas cercana. Devuelve (M, angulo_deg)."""
    mejor, mejor_ang = None, np.inf
    for M in rots:
        traza = np.trace(M.T @ delta)
        ang = np.degrees(np.arccos(np.clip((traza - 1.0) / 2.0, -1.0, 1.0)))
        if ang < mejor_ang:
            mejor, mejor_ang = M, ang
    return mejor, mejor_ang


def describe(M):
    partes = []
    for col in range(3):
        fila = int(np.argmax(np.abs(M[:, col])))
        signo = "+" if M[fila, col] > 0 else "-"
        partes.append(f"{'XYZ'[col]}->{signo}{'XYZ'[fila]}")
    return " ".join(partes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twh_bvh", default=str(DEFAULT_TWH))
    parser.add_argument("--lafan1_bvh", default=str(DEFAULT_LAFAN1))
    parser.add_argument("--step", type=int, default=5, help="Muestreo al buscar la pose.")
    parser.add_argument("--no_consensus", action="store_true", default=False,
                        help="No imponer coherencia por cadena (deja cada hueso suelto).")
    args = parser.parse_args()
    consenso = not args.no_consensus

    twh_frames, _ = load_twh_bvh_file(args.twh_bvh, verbose=False)
    laf_frames, _ = load_bvh_file(args.lafan1_bvh, format="lafan1")

    i_twh, s_twh = find_reference_frame(twh_frames, LANDMARKS_TWH, args.step, "TWH")
    i_laf, s_laf = find_reference_frame(laf_frames, LANDMARKS_LAFAN1, args.step, "LAFAN1")

    if min(s_twh["total"], s_laf["total"]) < 0.5:
        print("\n[yellow]AVISO: alguna pose de referencia es pobre. Las deltas "
              "resultantes seran menos fiables; revisa el angulo de ajuste.[/yellow]")

    f_twh, f_laf = twh_frames[i_twh], laf_frames[i_laf]
    rots = octahedral_rotations()

    print(f"\n[bold cyan]{'=' * 76}[/bold cyan]")
    print("[bold cyan]Delta por hueso: cruda -> ajustada al grupo octaedrico[/bold cyan]")
    print(f"[bold cyan]{'=' * 76}[/bold cyan]")

    deltas, angulos = {}, {}
    for link, (bone_laf, bone_twh) in BONE_MAP.items():
        if bone_laf not in f_laf or bone_twh not in f_twh:
            print(f"  [red]{link}: falta {bone_laf} o {bone_twh}[/red]")
            continue
        R_laf = R.from_quat(f_laf[bone_laf][1], scalar_first=True).as_matrix()
        R_twh = R.from_quat(f_twh[bone_twh][1], scalar_first=True).as_matrix()
        delta_cruda = R_twh.T @ R_laf

        M, ang = snap(delta_cruda, rots)
        deltas[link] = M
        angulos[link] = ang
        color = "green" if ang < 20 else ("yellow" if ang < 35 else "red")
        print(f"  {link:24s} {describe(M):26s} ajuste [{color}]{ang:5.1f} deg[/{color}]")

    print(f"\n[bold]Coherencia por cadena[/bold] "
          f"(los huesos de una cadena comparten convencion de rig):")
    for chain, links in CHAINS.items():
        presentes = [l for l in links if l in deltas]
        if not presentes:
            continue
        firmas = {describe(deltas[l]) for l in presentes}
        if len(firmas) == 1:
            print(f"  {chain:10s}: [green]coherente[/green]  {firmas.pop()}")
            continue

        print(f"  {chain:10s}: [yellow]{len(firmas)} convenciones distintas[/yellow]")
        for l in presentes:
            print(f"      {l:24s} {describe(deltas[l]):26s} ajuste {angulos[l]:5.1f} deg")

        if consenso:
            # El eje del hueso lo fija la direccion al hijo, pero el GIRO alrededor
            # de ese eje no: es el grado de libertad que el ajuste puede errar. Se
            # adopta la delta del miembro con menor angulo de ajuste, que es la
            # medida mas fiable de la cadena.
            lider = min(presentes, key=lambda l: angulos[l])
            for l in presentes:
                if l != lider:
                    deltas[l] = deltas[lider]
            print(f"      [cyan]consenso -> {describe(deltas[lider])} "
                  f"(de {lider}, ajuste {angulos[lider]:.1f} deg)[/cyan]")

    peor = max(angulos.values()) if angulos else 0.0
    print(f"\n  Angulo de ajuste maximo: {peor:.1f} deg "
          f"({'[green]metodo solido[/green]' if peor < 35 else '[red]correspondencia dudosa[/red]'})")

    # Escribir la config derivada
    with open(BASE_CONFIG) as f:
        config = json.load(f)
    out = copy.deepcopy(config)
    for table in ("ik_match_table1", "ik_match_table2"):
        for link, M in deltas.items():
            if link not in out[table]:
                continue
            entry = out[table][link]
            offset_lafan1 = R.from_quat(entry[4], scalar_first=True)
            nuevo = R.from_matrix(M) * offset_lafan1
            entry[4] = [float(v) for v in nuevo.as_quat(scalar_first=True)]

    out["_derivacion"] = {
        "metodo": "pose de referencia erguida + ajuste al grupo octaedrico",
        "consenso_por_cadena": consenso,
        "twh_bvh": pathlib.Path(args.twh_bvh).name,
        "twh_frame": int(i_twh),
        "lafan1_bvh": pathlib.Path(args.lafan1_bvh).name,
        "lafan1_frame": int(i_laf),
        "delta_por_link": {k: describe(v) for k, v in deltas.items()},
        "angulo_ajuste_deg": {k: round(v, 2) for k, v in angulos.items()},
    }

    with open(OUT_CONFIG, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4, ensure_ascii=False)
    print(f"\n[green]Escrito {OUT_CONFIG}[/green]")


if __name__ == "__main__":
    main()
