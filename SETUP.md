# Reproducir este repo en otra maquina

Guia para dejar el pipeline de retargeting TWH -> Unitree G1 funcionando desde cero.
Los comandos del dia a dia estan en [`comandos.md`](comandos.md).

## Lo que SI viene en el repo

- `general_motion_retargeting/` — libreria base (IK sobre mink/mujoco)
- `assets/` — modelos MuJoCo de los robots, incluido `unitree_g1`
- `scripts/` — scripts originales de GMR
- `scripts_inofuente/` — **el trabajo de la tesis**: loader del dataset TWH,
  script de retargeting, herramientas de diagnostico y calibracion
- `scripts_inofuente/ik_configs/` — las tres configs IK TWH -> G1
  (`bvh_twh_to_g1.json`, `_calibrated`, `_derived`)
- `snapshots/` — comparativas visuales de cada iteracion
- `requirements-lock.txt` — el entorno exacto que funciona

## Lo que NO viene (y hay que conseguir aparte)

Estan excluidos por `.gitignore` porque no caben en GitHub:

| Ruta | Tamano | Que es |
|---|---|---|
| `motion_data/twh_genea_challenge/` | ~40.2 GB | Dataset TWH / GENEA Challenge 2023 (896 BVH) |
| `motion_data/lafan1/` | ~333 MB | LAFAN1 (77 BVH), usado como baseline de comparacion |
| `.venv/` | — | Entorno virtual, se recrea |
| `retargeting_data/*.pkl` | — | Salidas del retargeting, se regeneran |
| `assets/body_models/` | — | Modelos SMPL-X, solo si usas el pipeline SMPL-X |

## 1. Clonar

```powershell
git clone https://github.com/KevinInoCol/Thesis-2026.git
cd Thesis-2026
```

## 2. Entorno de Python

El entorno original se creo con [uv](https://docs.astral.sh/uv/) y **Python 3.10**
(`python_requires>=3.10` en `setup.py`).

```powershell
uv venv --python 3.10
uv pip install -r requirements-lock.txt
uv pip install -e .
```

Si prefieres pip/conda clasico:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\activate.ps1
pip install -r requirements-lock.txt
pip install -e .
```

Nota: `requirements-lock.txt` incluye `smplx` desde git, asi que necesitas `git`
en el PATH durante la instalacion.

Activar en PowerShell (hay que poner el `.ps1` explicito; "activate" sin
extension resuelve a `activate.bat` y no afecta a la sesion):

```powershell
.\.venv\Scripts\activate.ps1
$env:PYTHONUTF8="1"
```

## 3. Datasets

### LAFAN1 (baseline, ~333 MB)

Descarga `lafan1.zip` del repo oficial de Ubisoft
([ubisoft-laforge-animation-dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset),
archivo `lafan1/lafan1.zip`) y extrae los 77 `.bvh` planos en:

```
motion_data/lafan1/*.bvh
```

### TWH / GENEA Challenge 2023 (~40.2 GB)

Los datos vienen del GENEA Challenge 2023, derivado de *Talking With Hands 16.2M*.
Requiere aceptar los terminos de uso del challenge — busca la release oficial en
el sitio del [GENEA Workshop](https://genea-workshop.github.io/) y descarga los
tres splits (`genea2023_trn`, `genea2023_val`, `genea2023_tst`).

> Anota aqui la URL exacta que usaste, para que la tesis sea citable:
> `TODO: pegar enlace de descarga`

La estructura que esperan los scripts es esta (verificada en la maquina original):

```
motion_data/twh_genea_challenge/
├── genea2023_trn/
│   └── genea2023_dataset/
│       └── trn/
│           ├── main-agent/
│           │   └── bvh/trn_2023_v0_000_main-agent.bvh ...
│           ├── interloctr/
│           └── metadata.csv
├── genea2023_val/   (misma forma, split "val")
└── genea2023_tst/   (misma forma, split "tst")
```

Es decir: cada zip se extrae en su propia carpeta `genea2023_<split>/`, y adentro
queda un `genea2023_dataset/<split>/main-agent/bvh/`. Si aplanas esa jerarquia,
las rutas de `comandos.md` dejan de funcionar.

Solo hacen falta los `.bvh` de `main-agent/` para el pipeline actual; el resto
(audio, TSV, `interloctr/`) es lo que infla el tamano y puedes omitirlo si solo
quieres reproducir el retargeting.

## 4. Verificar

```powershell
# Sanity check: visualizar un motion LAFAN1 ya retargeteado
python scripts\bvh_to_robot.py --bvh_file motion_data\lafan1\fight1_subject2.bvh --robot unitree_g1 --format lafan1 --rate_limit --save_path retargeting_data\g1_fight1.pkl

# El pipeline de la tesis: TWH -> G1 con la config derivada
python scripts_inofuente\twh_to_robot.py --bvh_file "motion_data\twh_genea_challenge\genea2023_trn\genea2023_dataset\trn\main-agent\bvh\trn_2023_v0_000_main-agent.bvh" --ik_config scripts_inofuente\ik_configs\bvh_twh_to_g1_derived.json --rate_limit
```

Si eso corre y abre el viewer de MuJoCo, el entorno esta bien.

Compara el resultado contra `snapshots/twh_final.png` para confirmar que la
calibracion se reprodujo igual.

## Origen del codigo

Este repo parte de [YanjieZe/GMR](https://github.com/YanjieZe/GMR) (commit
`bb1bbe4`, licencia MIT). Todo el historial upstream esta preservado, asi que
`git log scripts_inofuente/` aisla el trabajo propio de la tesis.
