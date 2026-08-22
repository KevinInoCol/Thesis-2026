"""
EXPERIMENTO DESCARTADO. Se conserva como registro negativo; NO usar su salida.

Por que se descarto
-------------------
Este script busca los rot_offset minimizando el RESIDUO DEL IK. Ese objetivo esta
mal: el residuo mide si el robot ALCANZA los objetivos, no si los objetivos son
anatomicamente CORRECTOS. Es un criterio de alcanzabilidad, no de correccion, y
premia configuraciones que generan objetivos comodos pero equivocados.

Sintoma que lo delato: bajo el residuo de 4.10 a 1.76 (2.33x) mientras la postura
del robot seguia rota (doblado 60 grados por la cintura) y la saturacion se
quedaba clavada en 13 DoF. La mejora del numero no correspondia a mejora real.

El metodo valido esta en derive_rot_offsets.py: resuelve el algebra directamente
sobre una pose de referencia erguida y ajusta al grupo octaedrico, validando con
capturas y saturacion en lugar de residuo.

Contexto original
-----------------
bvh_twh_to_g1.json es un clon de bvh_lafan1_to_g1.json con los nombres de huesos
remapeados y los pesos intactos. Esa linea base sale muy degradada (residuo IK
~5x peor, 17 DoF saturados, robot volcado en horizontal) porque los rot_offset
codifican la transformada del marco del hueso HUMANO al marco del link del
ROBOT, y TWH no comparte la convencion de ejes de LAFAN1:

    hueso          LAFAN1   TWH
    brazo          +X       +X     coincide
    antebrazo      +X       +X     coincide
    muslo          +X       -Y     distinto
    pierna         +X       -Y     distinto
    pelvis->espina +X       -Z     distinto
    espina->cuello +X       +Y     distinto

Metodo
------
La config de LAFAN1 funciona, luego para un hueso en correspondencia:

    R_robot = R_lafan1 * offset_lafan1                       (config que funciona)
    R_robot = R_twh    * offset_twh                          (lo que queremos)
    =>  offset_twh = (R_twh^-1 * R_lafan1) * offset_lafan1
                     \_____ Delta ______/

Delta es una rotacion constante por hueso que depende solo de la convencion de
los dos esqueletos. Como las diferencias medidas son de ~90 grados exactos,
Delta es con casi total seguridad una permutacion de ejes con signo, es decir un
elemento del grupo de rotaciones del octaedro: solo 24 candidatos.

Se buscan exhaustivamente esos 24 por CADENA (no por hueso suelto, que
multiplicaria el espacio sin sentido fisico: los huesos de una misma cadena
comparten convencion), evaluando el residuo IK real sobre una muestra de frames.
Las cadenas se calibran en orden de impacto decreciente: la pelvis es un free
joint y un error suyo rota el robot entero, asi que va primero.

Uso:
    python scripts_inofuente/calibrate_rot_offsets.py
    python scripts_inofuente/calibrate_rot_offsets.py --frames 30 --passes 2
"""

import argparse
import contextlib
import copy
import io
import itertools
import json
import pathlib
import sys

import mujoco as mj
import numpy as np
from rich import print
from scipy.spatial.transform import Rotation as R

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import params

from twh_loader import load_twh_bvh_file

BASE_CONFIG = HERE / "ik_configs" / "bvh_twh_to_g1.json"
OUT_CONFIG = HERE / "ik_configs" / "bvh_twh_to_g1_calibrated.json"
TMP_CONFIG = HERE / "ik_configs" / "_tmp_calibration.json"

DEFAULT_BVH = (
    HERE.parent / "motion_data" / "twh_genea_challenge" / "genea2023_trn"
    / "genea2023_dataset" / "trn" / "main-agent" / "bvh"
    / "trn_2023_v0_000_main-agent.bvh"
)

# Cadenas cinematicas que comparten convencion de ejes, en orden de impacto.
# La pelvis primero: es free joint y su error rota el robot completo.
CHAINS = {
    "pelvis": ["pelvis"],
    "torso": ["torso_link"],
    "piernas": [
        "left_hip_yaw_link", "left_knee_link", "left_ankle_roll_link",
        "right_hip_yaw_link", "right_knee_link", "right_ankle_roll_link",
    ],
    "brazos": [
        "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_yaw_link",
        "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_yaw_link",
    ],
}

SAT_TOL_RAD = np.deg2rad(2.0)


def octahedral_rotations():
    """Las 24 rotaciones propias que permutan los ejes con signo (det = +1)."""
    rots = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = signs[i]
            if np.isclose(np.linalg.det(M), 1.0):
                rots.append(M)
    assert len(rots) == 24, len(rots)
    return rots


def apply_delta(config, links, delta_matrix):
    """Devuelve una copia de config con Delta aplicado a los links indicados."""
    out = copy.deepcopy(config)
    delta = R.from_matrix(delta_matrix)
    for table in ("ik_match_table1", "ik_match_table2"):
        for link in links:
            if link not in out[table]:
                continue
            entry = out[table][link]
            offset_actual = R.from_quat(entry[4], scalar_first=True)
            nuevo = delta * offset_actual
            entry[4] = [float(v) for v in nuevo.as_quat(scalar_first=True)]
    return out


def evaluate(config, frames, human_height, robot):
    """Instancia GMR con la config dada y devuelve (residuo_medio, dof_saturados)."""
    with open(TMP_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f)
    params.IK_CONFIG_DICT["_calib"] = {robot: TMP_CONFIG}

    # GMR imprime DoF/bodies/motors de forma incondicional; lo silenciamos.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        retargeter = GMR(
            src_human="_calib", tgt_robot=robot,
            actual_human_height=human_height, verbose=False,
        )
        errs = np.zeros(len(frames))
        qpos_all = []
        for i, fr in enumerate(frames):
            qpos = retargeter.retarget(fr)
            qpos_all.append(qpos.copy())
            errs[i] = retargeter.error1()

    qpos_all = np.array(qpos_all)
    model = retargeter.model
    n_sat = 0
    for j in range(model.njnt):
        if model.jnt_type[j] != mj.mjtJoint.mjJNT_HINGE or not model.jnt_limited[j]:
            continue
        lo, hi = model.jnt_range[j]
        vals = qpos_all[:, model.jnt_qposadr[j]]
        if np.any(vals <= lo + SAT_TOL_RAD) or np.any(vals >= hi - SAT_TOL_RAD):
            n_sat += 1

    return float(errs.mean()), n_sat


def describe(M):
    """Etiqueta legible de una permutacion de ejes: a que va cada eje de entrada."""
    nombres = []
    for col in range(3):
        fila = int(np.argmax(np.abs(M[:, col])))
        signo = "+" if M[fila, col] > 0 else "-"
        nombres.append(f"{'XYZ'[col]}->{signo}{'XYZ'[fila]}")
    return " ".join(nombres)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh", default=str(DEFAULT_BVH))
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--frames", type=int, default=24,
                        help="Frames muestreados para evaluar cada candidato.")
    parser.add_argument("--passes", type=int, default=2,
                        help="Pasadas sobre las cadenas (la 2a refina con el resto ya fijado).")
    args = parser.parse_args()

    with open(BASE_CONFIG) as f:
        config = json.load(f)

    todos, human_height = load_twh_bvh_file(args.bvh, verbose=False)
    idx = np.linspace(0, len(todos) - 1, args.frames, dtype=int)
    frames = [todos[i] for i in idx]
    print(f"Muestra: {len(frames)} frames de {len(todos)}; altura {human_height:.3f} m")

    rots = octahedral_rotations()
    identidad = next(i for i, M in enumerate(rots) if np.allclose(M, np.eye(3)))

    err0, sat0 = evaluate(config, frames, human_height, args.robot)
    print(f"\n[bold]Linea base[/bold]: residuo {err0:.4f}, {sat0} DoF saturados\n")

    mejor_global = err0
    elegidos = {}
    # Delta ACUMULADA por cadena. Si una pasada aplica D1 y la siguiente D2, el
    # offset final es D2*D1*offset_original, asi que hay que multiplicar y no
    # sobreescribir: reportar solo la ultima pasada daria una config irreproducible.
    acumulado = {chain: np.eye(3) for chain in CHAINS}

    for p in range(args.passes):
        print(f"[bold cyan]--- Pasada {p + 1}/{args.passes} ---[/bold cyan]")
        for chain, links in CHAINS.items():
            resultados = []
            for k, M in enumerate(rots):
                cand = apply_delta(config, links, M)
                err, sat = evaluate(cand, frames, human_height, args.robot)
                resultados.append((err, sat, k))

            resultados.sort(key=lambda t: (t[0], t[1]))
            err, sat, k = resultados[0]

            if err < mejor_global - 1e-6:
                config = apply_delta(config, links, rots[k])
                acumulado[chain] = rots[k] @ acumulado[chain]
                marca = "[green]MEJORA[/green]"
                mejor_global = err
            else:
                err = next(e for e, s, kk in resultados if kk == identidad)
                sat = next(s for e, s, kk in resultados if kk == identidad)
                marca = "[dim]sin cambio[/dim]"

            acc = acumulado[chain]
            elegidos[chain] = (
                "identidad" if np.allclose(acc, np.eye(3)) else describe(acc)
            )
            print(f"  {chain:9s} -> {elegidos[chain]:28s} "
                  f"residuo {err:.4f}  sat {sat:2d}  {marca}")
        print()

    err_f, sat_f = evaluate(config, frames, human_height, args.robot)
    print(f"[bold]Resultado[/bold]: residuo {err0:.4f} -> {err_f:.4f} "
          f"({err0 / max(err_f, 1e-9):.2f}x mejor), "
          f"DoF saturados {sat0} -> {sat_f}")

    with open(OUT_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"[green]Escrito {OUT_CONFIG}[/green]")

    print("\n[bold]Delta por cadena:[/bold]")
    for chain, d in elegidos.items():
        print(f"  {chain:9s}: {d}")

    if TMP_CONFIG.exists():
        TMP_CONFIG.unlink()


if __name__ == "__main__":
    main()
