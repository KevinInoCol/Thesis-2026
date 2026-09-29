# Activar el entorno (en PowerShell hay que poner .ps1 explicitamente:
# "activate" sin extension resuelve a activate.bat y no afecta a la sesion)
cd C:\scripts-kevin\GMR
& .\.venv\Scripts\activate.ps1
$env:PYTHONUTF8="1"

# Ver un movimiento ya retargeteado (interactivo: espacio pausa, [ y ] cambian motion)
python scripts\vis_robot_motion.py --robot unitree_g1 --robot_motion_path retargeting_data\g1_dance1_subject1.pkl

# Retargetear otro, a velocidad real
python scripts\bvh_to_robot.py --bvh_file motion_data\lafan1\fight1_subject2.bvh --robot unitree_g1 --format lafan1 --rate_limit --save_path retargeting_data\g1_fight1.pkl

# Lote completo (sin visualización, más rápido)
python scripts\bvh_to_robot_dataset.py --src_folder motion_data\lafan1 --tgt_folder retargeting_data\g1_all --robot unitree_g1












cd C:\scripts-kevin\GMR
& .\.venv\Scripts\activate.ps1
$env:PYTHONUTF8="1"

# El resultado actual, animado
python scripts\vis_robot_motion.py --robot unitree_g1 --robot_motion_path retargeting_data\twh_final200.pkl

Son solo 200 frames (~7 s). Si quieres la secuencia completa de 65 s, genérala primero:

python scripts_inofuente\twh_to_robot.py --bvh_file "motion_data\twh_genea_challenge\genea2023_trn\genea2023_dataset\trn\main-agent\bvh\trn_2023_v0_000_main-agent.bvh" --ik_config scripts_inofuente\ik_configs\bvh_twh_to_g1_derived.json --rate_limit




- thesis → https://github.com/KevinInoCol/Thesis-2026 (el tuyo). Tu rama local master sigue a thesis/main y está al día: no hay cambios sin commitear ni commits sin subir.
- origin → https://github.com/YanjieZe/GMR, que es el proyecto original de GMR del que partiste, no el tuyo.