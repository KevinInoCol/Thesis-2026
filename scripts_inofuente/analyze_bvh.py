"""
Analizador estructural de ficheros BVH, para comparar el esqueleto de un dataset
nuevo (p.ej. Talking With Hands / GENEA Challenge) contra el que GMR ya soporta
(LAFAN1) y determinar exactamente que hace falta para anadir soporte.

Uso:
    python scripts_inofuente/analyze_bvh.py --bvh <fichero.bvh>
    python scripts_inofuente/analyze_bvh.py --bvh <nuevo.bvh> --ref <lafan1.bvh>
    python scripts_inofuente/analyze_bvh.py --bvh <nuevo.bvh> --ref <lafan1.bvh> --json informe.json
"""

import argparse
import json
import pathlib
import re

import numpy as np
from rich import print

# Huesos que el loader de GMR exige por nombre literal (utils/lafan1.py:33-37)
LOADER_REQUIRED = {
    "lafan1": ["LeftFoot", "LeftToe", "RightFoot", "RightToe"],
    "nokov": ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"],
}


def parse_hierarchy(path):
    """Recorre el bloque HIERARCHY y devuelve la lista de articulaciones.

    Cada entrada: dict(name, parent, depth, offset, channels, channel_names).
    Los End Site se ignoran, igual que hace el parser del upstream.
    """
    joints = []
    stack = []
    pending_end_site = False

    with open(path, "r") as f:
        for line in f:
            if "MOTION" in line:
                break

            rmatch = re.match(r"\s*ROOT\s+(\S+)", line)
            jmatch = re.match(r"\s*JOINT\s+(\S+)", line)
            if rmatch or jmatch:
                name = (rmatch or jmatch).group(1)
                parent = stack[-1] if stack else None
                joints.append(
                    {
                        "name": name,
                        "parent": parent,
                        "depth": len(stack),
                        "offset": None,
                        "channels": None,
                        "channel_names": None,
                    }
                )
                stack.append(name)
                continue

            if "End Site" in line:
                pending_end_site = True
                continue

            if "}" in line:
                if pending_end_site:
                    pending_end_site = False
                elif stack:
                    stack.pop()
                continue

            offmatch = re.match(
                r"\s*OFFSET\s+(-?[\d\.eE+-]+)\s+(-?[\d\.eE+-]+)\s+(-?[\d\.eE+-]+)", line
            )
            if offmatch and not pending_end_site and joints:
                joints[-1]["offset"] = [float(v) for v in offmatch.groups()]
                continue

            cmatch = re.match(r"\s*CHANNELS\s+(\d+)\s+(.*)", line)
            if cmatch and not pending_end_site and joints:
                joints[-1]["channels"] = int(cmatch.group(1))
                joints[-1]["channel_names"] = cmatch.group(2).split()
                continue

    return joints


def parse_motion_meta(path):
    """Lee Frames / Frame Time y la primera linea de datos para inspeccionar formato."""
    meta = {"frames": None, "frame_time": None, "fps": None, "first_data_line": None}
    in_motion = False

    with open(path, "r") as f:
        for line in f:
            if "MOTION" in line:
                in_motion = True
                continue
            if not in_motion:
                continue

            fmatch = re.match(r"\s*Frames:\s+(\d+)", line)
            if fmatch:
                meta["frames"] = int(fmatch.group(1))
                continue

            tmatch = re.match(r"\s*Frame Time:\s+([\d\.eE+-]+)", line)
            if tmatch:
                meta["frame_time"] = float(tmatch.group(1))
                meta["fps"] = round(1.0 / meta["frame_time"], 4)
                continue

            if line.strip():
                meta["first_data_line"] = line.rstrip("\n")
                break

    return meta


def diagnose_separator(data_line):
    """El parser del upstream hace line.strip().split(' ') -- separador rigido.

    Devuelve (n_valores_split_espacio_simple, n_valores_split_generico, es_compatible).
    """
    naive = data_line.strip().split(" ")
    generic = data_line.split()
    # split(' ') produce cadenas vacias si hay espacios dobles o tabuladores
    compatible = all(tok != "" for tok in naive) and len(naive) == len(generic)
    return len(naive), len(generic), compatible


def infer_scale_and_up(joints):
    """Infiere unidades a partir de la magnitud de los offsets del esqueleto.

    OJO: el eje de mayor extension NO es fiable como eje vertical. En LAFAN1 los
    miembros se extienden a lo largo de X en la pose de reposo, asi que esta
    heuristica dice "X" cuando el fichero es en realidad Y-up. El eje vertical
    solo se determina de forma fiable empiricamente (ver verify_bvh_geometry.py).
    """
    offsets = np.array([j["offset"] for j in joints if j["offset"] is not None])
    if len(offsets) == 0:
        return {}

    total_extent = np.abs(offsets).sum(axis=0)
    eje_mayor = "XYZ"[int(np.argmax(total_extent))]
    # La suma de |offset| a lo largo de la cadena aproxima la altura del sujeto
    max_chain = float(total_extent.max())

    if max_chain > 20:
        units = "centimetros (probable)"
        to_meters = 0.01
    elif max_chain > 0.5:
        units = "metros (probable)"
        to_meters = 1.0
    else:
        units = "indeterminado"
        to_meters = None

    return {
        "eje_de_mayor_extension": eje_mayor,  # NO es el eje vertical, ver docstring
        "extent_por_eje": [round(v, 3) for v in total_extent.tolist()],
        "unidades": units,
        "factor_a_metros": to_meters,
    }


def analyze(path, label):
    path = pathlib.Path(path)
    joints = parse_hierarchy(path)
    meta = parse_motion_meta(path)

    channel_counts = sorted({j["channels"] for j in joints if j["channels"]})
    uniform = len(channel_counts) == 1
    n_values_expected = sum(j["channels"] or 0 for j in joints)

    sep = None
    if meta["first_data_line"]:
        naive, generic, compatible = diagnose_separator(meta["first_data_line"])
        sep = {
            "valores_split_espacio_simple": naive,
            "valores_split_generico": generic,
            "compatible_con_parser_upstream": compatible,
            "valores_esperados_por_jerarquia": n_values_expected,
        }

    rot_order = None
    for j in joints:
        if j["channel_names"]:
            rots = [c for c in j["channel_names"] if "rotation" in c.lower()]
            rot_order = "".join(c[0].lower() for c in rots)
            break

    report = {
        "label": label,
        "fichero": path.name,
        "tamano_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "n_articulaciones": len(joints),
        "raiz": joints[0]["name"] if joints else None,
        "canales_por_articulacion": channel_counts,
        "canales_uniformes": uniform,
        "orden_rotacion": rot_order,
        "n_valores_por_frame": n_values_expected,
        "frames": meta["frames"],
        "frame_time": meta["frame_time"],
        "fps": meta["fps"],
        "duracion_seg": round(meta["frames"] * meta["frame_time"], 2)
        if meta["frames"] and meta["frame_time"]
        else None,
        "separador": sep,
        "escala": infer_scale_and_up(joints),
        "articulaciones": [j["name"] for j in joints],
        "jerarquia": [
            {"name": j["name"], "parent": j["parent"], "depth": j["depth"]} for j in joints
        ],
    }
    return report


def print_report(rep):
    print(f"\n[bold cyan]{'=' * 70}[/bold cyan]")
    print(f"[bold cyan]{rep['label']}: {rep['fichero']}[/bold cyan] ({rep['tamano_mb']} MB)")
    print(f"[bold cyan]{'=' * 70}[/bold cyan]")
    print(f"  Articulaciones      : {rep['n_articulaciones']}  (raiz: {rep['raiz']})")
    print(f"  Canales/articulacion: {rep['canales_por_articulacion']}  uniforme={rep['canales_uniformes']}")
    print(f"  Orden de rotacion   : {rep['orden_rotacion']}")
    print(f"  Valores por frame   : {rep['n_valores_por_frame']}")
    print(f"  Frames              : {rep['frames']}  @ {rep['fps']} fps  ({rep['duracion_seg']} s)")
    esc = rep["escala"]
    if esc:
        print(f"  Eje mayor extension : {esc['eje_de_mayor_extension']} (NO es el eje vertical)  extent={esc['extent_por_eje']}")
        print(f"  Unidades            : {esc['unidades']}  (factor a metros: {esc['factor_a_metros']})")
    sep = rep["separador"]
    if sep:
        estado = "[green]OK[/green]" if sep["compatible_con_parser_upstream"] else "[red]INCOMPATIBLE[/red]"
        print(f"  Separador de datos  : {estado}  "
              f"(split(' ')={sep['valores_split_espacio_simple']}, "
              f"split()={sep['valores_split_generico']}, "
              f"esperados={sep['valores_esperados_por_jerarquia']})")


def print_tree(rep, max_depth=None):
    print(f"\n[bold]Jerarquia de {rep['label']}:[/bold]")
    for j in rep["jerarquia"]:
        if max_depth is not None and j["depth"] > max_depth:
            continue
        print(f"  {'  ' * j['depth']}{j['name']}")


def compare(new_rep, ref_rep):
    print(f"\n[bold yellow]{'=' * 70}[/bold yellow]")
    print(f"[bold yellow]COMPARATIVA: {new_rep['label']} vs {ref_rep['label']}[/bold yellow]")
    print(f"[bold yellow]{'=' * 70}[/bold yellow]")

    new_set = set(new_rep["articulaciones"])
    ref_set = set(ref_rep["articulaciones"])

    print(f"  Articulaciones      : {len(new_set)} vs {len(ref_set)}")
    print(f"  Nombres compartidos : {len(new_set & ref_set)}")
    if new_set & ref_set:
        print(f"    -> {sorted(new_set & ref_set)}")
    print(f"  Canales             : {new_rep['canales_por_articulacion']} vs {ref_rep['canales_por_articulacion']}")
    print(f"  FPS                 : {new_rep['fps']} vs {ref_rep['fps']}")
    print(f"  Orden de rotacion   : {new_rep['orden_rotacion']} vs {ref_rep['orden_rotacion']}")

    print("\n[bold]Huesos que el loader de GMR exige por nombre literal:[/bold]")
    for fmt, required in LOADER_REQUIRED.items():
        faltan = [b for b in required if b not in new_set]
        estado = "[green]presentes[/green]" if not faltan else f"[red]faltan {faltan}[/red]"
        print(f"  --format {fmt:8s}: {estado}")

    print("\n[bold]Huesos del ik_config bvh_lafan1_to_g1.json y su estado en el dataset nuevo:[/bold]")
    ik_bones = [
        "Hips", "Spine2", "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg",
        "LeftFootMod", "RightFootMod", "LeftArm", "RightArm",
        "LeftForeArm", "RightForeArm", "LeftHand", "RightHand",
    ]
    for b in ik_bones:
        if b.endswith("Mod"):
            print(f"  {b:16s}: [yellow]sintetico (lo construye el loader)[/yellow]")
        elif b in new_set:
            print(f"  {b:16s}: [green]presente[/green]")
        else:
            print(f"  {b:16s}: [red]ausente -> hay que remapear[/red]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh", required=True, help="BVH a analizar (dataset nuevo).")
    parser.add_argument("--ref", default=None, help="BVH de referencia (p.ej. LAFAN1).")
    parser.add_argument("--tree", action="store_true", help="Imprime la jerarquia completa.")
    parser.add_argument("--max-depth", type=int, default=None, help="Profundidad max del arbol.")
    parser.add_argument("--json", default=None, help="Guarda el informe en JSON.")
    args = parser.parse_args()

    new_rep = analyze(args.bvh, "NUEVO")
    print_report(new_rep)
    if args.tree:
        print_tree(new_rep, args.max_depth)

    ref_rep = None
    if args.ref:
        ref_rep = analyze(args.ref, "REFERENCIA")
        print_report(ref_rep)
        if args.tree:
            print_tree(ref_rep, args.max_depth)
        compare(new_rep, ref_rep)

    if args.json:
        out = {"nuevo": new_rep, "referencia": ref_rep}
        pathlib.Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nInforme guardado en {args.json}")


if __name__ == "__main__":
    main()
