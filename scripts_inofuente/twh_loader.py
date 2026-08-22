"""
Loader BVH para Talking With Hands / GENEA Challenge (esqueleto TWH de 83 huesos).

Equivalente a general_motion_retargeting/utils/lafan1.py pero para el esqueleto
TWH. No modifica nada del upstream: reutiliza su parser (read_bvh maneja
correctamente los 6 canales por articulacion de TWH) y solo cambia el mapeo de
nombres y la sintesis del pseudo-hueso del pie.

Diferencias frente a LAFAN1 que motivan este fichero:
  - 83 articulaciones vs 22, y CERO nombres en comun.
  - Jerarquia con un nivel extra: body_world -> b_root. La pelvis es b_root;
    body_world es un contenedor en espacio de mundo.
  - 4 huesos de espina (b_spine0..3) frente a 3 en LAFAN1. El equivalente de
    "Spine2" (donde cuelgan hombros y cuello) es b_spine3.
  - NO hay huesos de dedos del pie. LAFAN1 tiene LeftFoot -> LeftToe, y el
    loader del upstream construye "LeftFootMod" con la posicion del pie mas la
    orientacion del toe. En TWH b_l_foot es terminal, asi que sintetizamos un
    toe virtual a partir del OFFSET de su End Site.

Coordenadas: TWH es Y-up y en centimetros, igual que LAFAN1, por lo que se
reutiliza tal cual la transformada del upstream (Y-up -> Z-up, cm -> m).
Verificado empiricamente: tras la transformada, b_root queda a z~0.90 m,
la cabeza a z~1.50 m y los pies a z~0.09 m.
"""

import re

import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh

# Correspondencia TWH -> LAFAN1, solo como documentacion del criterio seguido.
# El loader emite nombres TWH; es bvh_twh_to_g1.json quien los consume.
TWH_TO_LAFAN1 = {
    "b_root": "Hips",
    "b_spine3": "Spine2",
    "b_l_upleg": "LeftUpLeg",
    "b_r_upleg": "RightUpLeg",
    "b_l_leg": "LeftLeg",
    "b_r_leg": "RightLeg",
    "b_l_foot": "LeftFoot",
    "b_r_foot": "RightFoot",
    "b_l_arm": "LeftArm",
    "b_r_arm": "RightArm",
    "b_l_forearm": "LeftForeArm",
    "b_r_forearm": "RightForeArm",
    "b_l_wrist": "LeftHand",
    "b_r_wrist": "RightHand",
    "b_neck0": "Neck",
    "b_head": "Head",
    # sinteticos, construidos por este loader:
    "b_l_foot_mod": "LeftFootMod",
    "b_r_foot_mod": "RightFootMod",
    "b_l_toe": "LeftToe",
    "b_r_toe": "RightToe",
}

# Pies cuyo End Site usamos para sintetizar el toe virtual
FEET = ("b_l_foot", "b_r_foot")

# Estimacion de estatura.
#
# Primer intento: altura del hueso de la cabeza + un margen de coronilla fijo.
# Daba 1.595 m, y las proporciones resultantes delataban que era baja: dejaba la
# pelvis al 56.6% de la estatura (lo tipico en adultos es 52-55%) y el hueso de
# la cabeza al 93.7% (borde alto del rango 90-94%).
#
# Metodo actual: derivar la estatura de la altura de la pelvis, que es un
# landmark mucho mas estable que la cabeza (no depende de donde el rigger
# coloco el hueso dentro del craneo ni de la inclinacion del cuello).
# Con 0.535 la estatura sale ~1.69 m y la altura de pelvis del robot pasa de
# 0.756 a 0.787 m, frente a los 0.802 m de la linea base de LAFAN1.
PELVIS_RATIO = 0.535  # altura de pelvis / estatura, adulto de pie
CROWN_ALLOWANCE_M = 0.10  # solo para el contraste informativo


def parse_end_site_offsets(bvh_file):
    """Devuelve {nombre_articulacion_padre: offset_del_End_Site}.

    read_bvh del upstream descarta los End Site, pero en TWH el End Site del pie
    es la unica informacion disponible sobre la direccion de la planta, asi que
    la necesitamos para sintetizar el toe.
    """
    offsets = {}
    stack = []
    last_joint = None
    in_end_site = False

    with open(bvh_file, "r") as f:
        for line in f:
            if "MOTION" in line:
                break

            m = re.match(r"\s*(?:ROOT|JOINT)\s+(\S+)", line)
            if m:
                last_joint = m.group(1)
                stack.append(last_joint)
                continue

            if "End Site" in line:
                in_end_site = True
                continue

            if "}" in line:
                if in_end_site:
                    in_end_site = False
                elif stack:
                    stack.pop()
                continue

            om = re.match(
                r"\s*OFFSET\s+(-?[\d\.eE+-]+)\s+(-?[\d\.eE+-]+)\s+(-?[\d\.eE+-]+)", line
            )
            if om and in_end_site and stack:
                offsets[stack[-1]] = np.array([float(v) for v in om.groups()])
                continue

    return offsets


def load_twh_bvh_file(bvh_file, human_height=None, verbose=False):
    """Carga un BVH de TWH/GENEA y devuelve (frames, human_height).

    Cada frame es un dict {nombre_hueso_twh: [posicion_xyz_m, cuaternion_wxyz]},
    en el mismo formato que espera GeneralMotionRetargeting.
    """
    data = read_bvh(bvh_file)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)
    global_quats, global_pos = global_data[0], global_data[1]

    # Misma transformada que el upstream: Y-up -> Z-up
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    bone_idx = {bone: i for i, bone in enumerate(data.bones)}

    faltan = [b for b in FEET if b not in bone_idx]
    if faltan:
        raise ValueError(
            f"El BVH no parece ser del esqueleto TWH: faltan los huesos {faltan}. "
            f"Raiz encontrada: {data.bones[0]!r}, {len(data.bones)} articulaciones."
        )

    # Offsets de End Site de los pies, para el toe virtual (en espacio BVH, cm)
    end_sites = parse_end_site_offsets(bvh_file)
    toe_offsets = {}
    for foot in FEET:
        if foot in end_sites:
            toe_offsets[foot] = end_sites[foot]
        else:
            # Sin End Site no podemos orientar la planta; degradamos a vector nulo
            toe_offsets[foot] = np.zeros(3)
            if verbose:
                print(f"[twh_loader] AVISO: {foot} sin End Site, toe virtual degradado")

    if verbose:
        for foot, off in toe_offsets.items():
            print(f"[twh_loader] toe virtual de {foot}: offset BVH {off} cm")

    n_frames = global_pos.shape[0]
    frames = []

    for frame in range(n_frames):
        result = {}
        for bone, i in bone_idx.items():
            orientation = utils.quat_mul(rotation_quat, global_quats[frame, i])
            position = global_pos[frame, i] @ rotation_matrix.T / 100  # cm -> m
            result[bone] = [position, orientation]

        # Toe virtual: se propaga el offset del End Site por la rotacion global
        # del pie, en espacio BVH, y despues se transforma igual que el resto.
        for foot in FEET:
            side = "l" if "_l_" in foot else "r"
            i = bone_idx[foot]
            q_foot_bvh = global_quats[frame, i]
            rot_foot_bvh = R.from_quat(q_foot_bvh, scalar_first=True)
            tip_bvh = global_pos[frame, i] + rot_foot_bvh.apply(toe_offsets[foot])
            tip_pos = tip_bvh @ rotation_matrix.T / 100

            # El toe virtual no tiene canales de rotacion propios, asi que hereda
            # la orientacion global del pie (un hijo sin rotacion local).
            result[f"b_{side}_toe"] = [tip_pos, result[foot][1]]

            # Equivalente de LeftFootMod: posicion del tobillo + orientacion de
            # la planta. En LAFAN1 la orientacion viene del hueso del toe, que es
            # una articulacion real; aqui el toe es virtual y hereda la del pie,
            # asi que esto es exactamente la orientacion del pie.
            result[f"b_{side}_foot_mod"] = [result[foot][0], result[foot][1]]

        frames.append(result)

    if human_height is None:
        # Se estima del frame 0 a partir de la altura de la pelvis, landmark mas
        # estable que la cabeza. Ver la nota de PELVIS_RATIO.
        pelvis_z = frames[0]["b_root"][0][2]
        head_z = frames[0]["b_head"][0][2]
        human_height = float(pelvis_z / PELVIS_RATIO)
        if verbose:
            por_cabeza = head_z + CROWN_ALLOWANCE_M
            print(
                f"[twh_loader] altura estimada: {human_height:.3f} m "
                f"(pelvis z={pelvis_z:.3f} / {PELVIS_RATIO})"
            )
            print(
                f"[twh_loader]   contraste por cabeza: {por_cabeza:.3f} m "
                f"(descartado, subestimaba)"
            )

    return frames, human_height
