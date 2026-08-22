# Reproducir la transferencia BVH -> Unitree G1 en otra maquina

El objetivo de este repo es **demostrar el retargeting de movimiento humano
capturado en BVH hacia el robot Unitree G1**. No hace falta entrenar nada ni
descargar datasets completos: traes tus propios `.bvh` y le apuntas el script.

Los comandos del dia a dia estan en [`comandos.md`](comandos.md).

## 1. Clonar

```powershell
git clone https://github.com/KevinInoCol/Thesis-2026.git
cd Thesis-2026
```

## 2. Entorno de Python

Se creo con [uv](https://docs.astral.sh/uv/) y **Python 3.10**
(`python_requires>=3.10` en `setup.py`).

```powershell
uv venv --python 3.10
uv pip install -r requirements-lock.txt
uv pip install -e .
```

Con pip clasico:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\activate.ps1
pip install -r requirements-lock.txt
pip install -e .
```

`requirements-lock.txt` instala `smplx` desde git, asi que necesitas `git` en el
PATH durante la instalacion.

Para cada sesion nueva en PowerShell (el `.ps1` explicito es necesario:
"activate" sin extension resuelve a `activate.bat` y no afecta a la sesion):

```powershell
.\.venv\Scripts\activate.ps1
$env:PYTHONUTF8="1"
```

## 3. Trae tus BVH

No hay una carpeta obligatoria: `--bvh_file` acepta cualquier ruta. Puedes
dejarlos donde quieras, por ejemplo `motion_data/mis_bvh/` (esa carpeta esta en
`.gitignore`, asi que no se subiran al repo por accidente).

## 4. Correr la transferencia

```powershell
python scripts_inofuente\twh_to_robot.py `
  --bvh_file "ruta\a\tu_archivo.bvh" `
  --ik_config scripts_inofuente\ik_configs\bvh_twh_to_g1_derived.json `
  --rate_limit
```

Se abre el viewer de MuJoCo con el humano y el G1 lado a lado.

Opciones utiles (todas en `twh_to_robot.py`):

| Flag | Para que |
|---|---|
| `--max_frames 200` | Prueba rapida, solo los primeros N frames |
| `--save_path out.pkl` | Guarda el movimiento del robot |
| `--record_video --video_path out.mp4` | Graba video |
| `--headless` | Sin ventana (combinar con `--record_video`) |
| `--human_height 1.75` | Fija la estatura en vez de estimarla del frame 0 |
| `--motion_fps 30` | FPS del BVH (30 por defecto, que es lo de GENEA 2023) |

Para volver a ver un `.pkl` ya generado:

```powershell
python scripts\vis_robot_motion.py --robot unitree_g1 --robot_motion_path out.pkl
```

## 5. Cual de las tres configs IK usar

Estan en `scripts_inofuente/ik_configs/`, en orden de como se fueron obteniendo:

| Config | Que es |
|---|---|
| `bvh_twh_to_g1.json` | Linea base, mapeo directo TWH -> G1 sin corregir |
| `bvh_twh_to_g1_calibrated.json` | Offsets de rotacion ajustados con `calibrate_rot_offsets.py` |
| `bvh_twh_to_g1_derived.json` | **La que da mejor resultado.** Offsets derivados analiticamente con `derive_rot_offsets.py` |

Usa `_derived` salvo que quieras reproducir la comparativa. Las diferencias
visuales entre las tres estan en `snapshots/` (`twh_baseline.png`,
`twh_calibrado.png`, `twh_derivado.png`, `twh_final.png`).

## Importante: el esqueleto de tus BVH

El loader (`scripts_inofuente/twh_loader.py`) esta escrito para el **esqueleto
TWH / GENEA Challenge**: 83 articulaciones, raiz `body_world -> b_root`, nombres
tipo `b_l_upleg`, `b_spine3`, `b_l_wrist`. Tambien asume **Y-up y centimetros**.

Si el BVH no trae esos huesos, el loader falla a proposito con un mensaje claro
(`twh_loader.py:139`):

```
El BVH no parece ser del esqueleto TWH: faltan los huesos ['b_l_foot', 'b_r_foot'].
Raiz encontrada: 'Hips', 22 articulaciones.
```

Entonces:

- **BVH de GENEA / Talking With Hands** -> funciona tal cual. Es el caso probado.
- **BVH de LAFAN1** (raiz `Hips`, 22 huesos) -> usa el script del upstream, que
  ya lo soporta:
  ```powershell
  python scripts\bvh_to_robot.py --bvh_file tu.bvh --robot unitree_g1 --format lafan1 --rate_limit
  ```
- **BVH de otro rig** (Mixamo, OptiTrack, Vicon, Blender...) -> hay que adaptarlo.
  Para eso estan las herramientas de `scripts_inofuente/`, en el orden en que se
  usaron para TWH:

  1. `analyze_bvh.py` — inspecciona jerarquia, nombres, unidades y eje up
  2. `diagnose_bone_axes.py` — averigua como estan orientados los huesos
  3. `diagnose_scale.py` — verifica cm vs m y las proporciones del sujeto
  4. `make_twh_ik_config.py` — genera una config IK nueva a partir del mapeo
  5. `derive_rot_offsets.py` — deriva los offsets de rotacion
  6. `validate_retarget.py` / `inspect_joint_traj.py` — comprueba el resultado
  7. `render_snapshots.py` — genera las comparativas visuales

  En la practica hay que tocar dos cosas: el diccionario de nombres del loader y
  la config IK. `twh_loader.py` documenta en su docstring exactamente que hubo
  que resolver para TWH (jerarquia con nivel extra, 4 huesos de espina, ausencia
  de huesos de dedos del pie), que es la misma lista de problemas que aparece con
  cualquier rig nuevo.

## Baseline de comparacion (opcional)

`snapshots/lafan1_baseline.png` es la referencia contra la que se compararon los
resultados de TWH. Para regenerarla necesitas LAFAN1, que se descarga de
[ubisoft-laforge-animation-dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
(`lafan1/lafan1.zip`, ~333 MB) y se extrae en `motion_data/lafan1/`.

Cuando hace falta LAFAN1:

- **No** para transferir BVH de esqueleto TWH que ya funcionan.
- **Sí** para rehacer la comparativa de `snapshots/`.
- **Sí** para adaptar un rig nuevo: `derive_rot_offsets.py` y
  `validate_retarget.py` lo usan como referencia, y `diagnose_bone_axes.py`
  tiene la ruta fija en el codigo.

## Lo que no viene en el repo

Excluido por `.gitignore`, y ninguno hace falta para la transferencia:

- `.venv/` — se recrea con el paso 2
- `motion_data/` — tus BVH y cualquier dataset
- `retargeting_data/*.pkl` — salidas, se regeneran
- `assets/body_models/` — modelos SMPL-X, solo para el pipeline SMPL-X del upstream
- `videos/` — grabaciones

## Origen del codigo

Parte de [YanjieZe/GMR](https://github.com/YanjieZe/GMR) (commit `bb1bbe4`,
licencia MIT), con el historial upstream preservado. `git log scripts_inofuente/`
aisla el trabajo propio de la tesis.
