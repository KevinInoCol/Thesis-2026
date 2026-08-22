# Comandos

Referencia práctica. Contexto y detalles en [`SETUP.md`](SETUP.md).

---

## 0. Instalación en una máquina nueva (una sola vez)

```powershell
git clone https://github.com/KevinInoCol/Thesis-2026.git
cd Thesis-2026

# Entorno con uv (Python 3.10)
uv venv --python 3.10
uv pip install -r requirements-lock.txt
uv pip install -e .
```

Con pip clásico en lugar de uv:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\activate.ps1
pip install -r requirements-lock.txt
pip install -e .
```

Requiere `git` en el PATH (una dependencia, `smplx`, se instala desde git).

---

## 1. Activar el entorno (cada sesión)

En PowerShell hay que poner el `.ps1` explícitamente: "activate" sin extensión
resuelve a `activate.bat` y no afecta a la sesión.

```powershell
.\.venv\Scripts\activate.ps1
$env:PYTHONUTF8="1"
```

En la máquina de desarrollo original el repo está en `C:\scripts-kevin\GMR`:

```powershell
cd C:\scripts-kevin\GMR
& .\.venv\Scripts\activate.ps1
$env:PYTHONUTF8="1"
```

---

## 2. Transferencia BVH -> Unitree G1 (el pipeline de la tesis)

Apunta `--bvh_file` a cualquier BVH del esqueleto TWH / GENEA. La ruta puede ser
cualquiera; no hay carpeta obligatoria.

```powershell
python scripts_inofuente\twh_to_robot.py `
  --bvh_file "ruta\a\tu_archivo.bvh" `
  --ik_config scripts_inofuente\ik_configs\bvh_twh_to_g1_derived.json `
  --rate_limit
```

Prueba rápida, solo los primeros 200 frames (~7 s):

```powershell
python scripts_inofuente\twh_to_robot.py --bvh_file "ruta\a\tu_archivo.bvh" --ik_config scripts_inofuente\ik_configs\bvh_twh_to_g1_derived.json --max_frames 200 --save_path retargeting_data\prueba200.pkl
```

Guardando el resultado y grabando video:

```powershell
python scripts_inofuente\twh_to_robot.py --bvh_file "ruta\a\tu_archivo.bvh" --ik_config scripts_inofuente\ik_configs\bvh_twh_to_g1_derived.json --save_path retargeting_data\salida.pkl --record_video --video_path videos\salida.mp4
```

Flags: `--max_frames N`, `--save_path`, `--record_video --video_path`,
`--headless`, `--human_height 1.75`, `--motion_fps 30`, `--robot unitree_g1`.

### Las tres configs IK

| Config | Qué es |
|---|---|
| `bvh_twh_to_g1.json` | Línea base, mapeo directo sin corregir |
| `bvh_twh_to_g1_calibrated.json` | Offsets ajustados con `calibrate_rot_offsets.py` |
| `bvh_twh_to_g1_derived.json` | **La mejor.** Offsets derivados con `derive_rot_offsets.py` |

---

## 3. Volver a ver un resultado guardado

Interactivo: espacio pausa, `[` y `]` cambian de motion.

```powershell
python scripts\vis_robot_motion.py --robot unitree_g1 --robot_motion_path retargeting_data\salida.pkl
```

---

## 4. Diagnóstico, cuando un BVH no carga

El loader exige el esqueleto TWH (83 huesos, `b_root`, `b_l_foot`…, Y-up, cm). Si
falla con "El BVH no parece ser del esqueleto TWH", empieza por aquí.

Ojo: cada script usa un nombre de flag distinto, no están unificados.

```powershell
# Jerarquía, nombres de huesos, unidades y eje up.   Flag: --bvh
python scripts_inofuente\analyze_bvh.py --bvh "ruta\a\tu_archivo.bvh" --tree

# Igual, pero comparando contra un BVH de referencia y guardando el informe
python scripts_inofuente\analyze_bvh.py --bvh "ruta\a\tu_archivo.bvh" --ref motion_data\lafan1\walk1_subject1.bvh --json informe.json

# cm vs m y proporciones del sujeto.                 Flag: --twh_bvh
python scripts_inofuente\diagnose_scale.py --twh_bvh "ruta\a\tu_archivo.bvh"
```

`diagnose_bone_axes.py` **no acepta argumentos**: tiene las rutas fijas en el
código (`diagnose_bone_axes.py:29-30`) y compara LAFAN1 contra TWH. Para usarlo
con otro BVH hay que editar esas dos constantes.

### Adaptar un rig nuevo

```powershell
# 1. Verificar el mapeo de la config IK sin escribir nada
python scripts_inofuente\make_twh_ik_config.py --check

# 2. Derivar los offsets de rotación (necesita un BVH de LAFAN1 como referencia)
python scripts_inofuente\derive_rot_offsets.py --twh_bvh "ruta\a\tu_archivo.bvh" --lafan1_bvh motion_data\lafan1\walk1_subject1.bvh

# 3. Alternativa: calibrarlos por optimización.     Flag: --bvh
python scripts_inofuente\calibrate_rot_offsets.py --bvh "ruta\a\tu_archivo.bvh" --frames 24 --passes 2

# 4. Validar contra la línea base.                  Flags: --twh / --lafan1
python scripts_inofuente\validate_retarget.py --twh "ruta\a\tu_archivo.bvh" --lafan1 motion_data\lafan1\walk1_subject1.bvh --ik_config scripts_inofuente\ik_configs\bvh_twh_to_g1_derived.json

# 5. Revisar trayectorias articulares de un resultado
python scripts_inofuente\inspect_joint_traj.py --pkl retargeting_data\salida.pkl

# 6. Generar comparativas visuales
python scripts_inofuente\render_snapshots.py --pkl retargeting_data\salida.pkl --out snapshots\nuevo.png
```

**Importante:** los pasos 2 y 4 usan LAFAN1 como referencia, así que para
adaptar un rig nuevo sí necesitas LAFAN1 descargado (ver `SETUP.md`). Para
transferir BVH de esqueleto TWH que ya funcionan, no.

---

## 5. Pipeline LAFAN1 del upstream (baseline de comparación)

Requiere LAFAN1 en `motion_data\lafan1\` (opcional, ver `SETUP.md`).

```powershell
# Un motion, a velocidad real
python scripts\bvh_to_robot.py --bvh_file motion_data\lafan1\fight1_subject2.bvh --robot unitree_g1 --format lafan1 --rate_limit --save_path retargeting_data\g1_fight1.pkl

# Lote completo, sin visualización (más rápido)
python scripts\bvh_to_robot_dataset.py --src_folder motion_data\lafan1 --tgt_folder retargeting_data\g1_all --robot unitree_g1
```
