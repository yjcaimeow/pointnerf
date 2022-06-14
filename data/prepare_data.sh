srun --partition=VA-Human -n1 --ntasks-per-node 1 --gres=gpu:1 --job-name=tb_ft python ./load_waymo.py
