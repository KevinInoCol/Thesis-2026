"""
Genera scripts_inofuente/ik_configs/bvh_twh_to_g1.json a partir de la config del
upstream bvh_lafan1_to_g1.json.

El objetivo es una LINEA BASE HONESTA: se renombran unicamente los nombres de
huesos humanos (LAFAN1 -> TWH) y NO se toca ningun peso, offset ni cuaternion.
Asi cualquier mejora posterior de tuning es medible contra esta referencia.

Uso:
    python scripts_inofuente/make_twh_ik_config.py
    python scripts_inofuente/make_twh_ik_config.py --check   # solo verifica
"""

import argparse
import json
import pathlib

from rich import print

from general_motion_retargeting.params import IK_CONFIG_ROOT

HERE = pathlib.Path(__file__).parent
OUT_PATH = HERE / "ik_configs" / "bvh_twh_to_g1.json"
SRC_PATH = IK_CONFIG_ROOT / "bvh_lafan1_to_g1.json"

# LAFAN1 -> TWH. Ver el razonamiento de cada correspondencia en twh_loader.py.
RENAME = {
    "Hips": "b_root",
    "Spine2": "b_spine3",
    "LeftUpLeg": "b_l_upleg",
    "RightUpLeg": "b_r_upleg",
    "LeftLeg": "b_l_leg",
    "RightLeg": "b_r_leg",
    "LeftFootMod": "b_l_foot_mod",
    "RightFootMod": "b_r_foot_mod",
    "LeftArm": "b_l_arm",
    "RightArm": "b_r_arm",
    "LeftForeArm": "b_l_forearm",
    "RightForeArm": "b_r_forearm",
    "LeftHand": "b_l_wrist",
    "RightHand": "b_r_wrist",
}


def build():
    with open(SRC_PATH) as f:
        cfg = json.load(f)

    sin_mapear = set()

    def ren(name):
        if name in RENAME:
            return RENAME[name]
        sin_mapear.add(name)
        return name

    out = dict(cfg)
    out["human_root_name"] = ren(cfg["human_root_name"])
    out["human_scale_table"] = {ren(k): v for k, v in cfg["human_scale_table"].items()}

    for table in ("ik_match_table1", "ik_match_table2"):
        nuevo = {}
        for robot_link, entry in cfg[table].items():
            entry = list(entry)
            entry[0] = ren(entry[0])  # entry[0] es el nombre del hueso humano
            nuevo[robot_link] = entry
        out[table] = nuevo

    return cfg, out, sin_mapear


def verify(src, out):
    """Comprueba que solo cambiaron nombres: pesos y offsets identicos."""
    problemas = []

    for k in ("ground_height", "human_height_assumption", "robot_root_name",
              "use_ik_match_table1", "use_ik_match_table2"):
        if src[k] != out[k]:
            problemas.append(f"{k} cambio: {src[k]} -> {out[k]}")

    if sorted(src["human_scale_table"].values()) != sorted(out["human_scale_table"].values()):
        problemas.append("los valores de human_scale_table no coinciden")

    for table in ("ik_match_table1", "ik_match_table2"):
        if set(src[table].keys()) != set(out[table].keys()):
            problemas.append(f"{table}: los links del robot no coinciden")
            continue
        for link in src[table]:
            a, b = src[table][link], out[table][link]
            # entry = [hueso_humano, peso_pos, peso_rot, offset_pos, offset_rot]
            if a[1:] != b[1:]:
                problemas.append(f"{table}/{link}: pesos u offsets alterados")

    return problemas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Solo verifica, no escribe.")
    args = parser.parse_args()

    src, out, sin_mapear = build()

    print(f"Origen : {SRC_PATH}")
    print(f"Destino: {OUT_PATH}")

    if sin_mapear:
        print(f"[red]Huesos sin correspondencia en RENAME: {sorted(sin_mapear)}[/red]")
        print("[red]Se han dejado con el nombre de LAFAN1 y el IK fallara con ellos.[/red]")
    else:
        print("[green]Todos los huesos humanos tienen correspondencia TWH.[/green]")

    problemas = verify(src, out)
    if problemas:
        print("[red]La config NO es un clon fiel:[/red]")
        for p in problemas:
            print(f"  [red]- {p}[/red]")
    else:
        print("[green]Verificado: pesos, offsets y cuaterniones identicos al upstream.[/green]")

    n_links = len(out["ik_match_table1"])
    print(f"Links del robot mapeados: {n_links} en ik_match_table1, "
          f"{len(out['ik_match_table2'])} en ik_match_table2")

    if args.check:
        print("\nModo --check: no se ha escrito nada.")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4, ensure_ascii=False)
    print(f"\n[green]Escrito {OUT_PATH}[/green]")


if __name__ == "__main__":
    main()
