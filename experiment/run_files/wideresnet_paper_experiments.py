import subprocess

# python networks/eq_wrn.py -m model.kernel_layout=[3,3],[3,1],[1,3],[3,1,3],[1,3,1],[3,1,1]
# kernel_layout = [3,3],[3,1],[1,3],[3,1,3],[1,3,1],[3,1,1]


# experiment 1 Type of convolutions in residual block
# ran all
args_all = ["model=wrn,eq_wrn", "model.restrict=invariant"]
args = ["model.kernel_layout=[3,3]", "model.depth=28", "model.widen_factor=2", "wandb.notes=exp_1_wrn"]
subprocess.run(["python", "../main.py", "-m", args_all[0], args_all[1], args[0], args[1], args[2], args[3]])

args = ["model.kernel_layout=[1,3,1]", "model.depth=40", "model.widen_factor=2", "wandb.notes=exp_1_wrn"]
subprocess.run(["python", "../main.py", "-m", args_all[0], args_all[1], args[0], args[1], args[2], args[3]])

args = ["model.kernel_layout=[3,1]", "model.depth=40", "model.widen_factor=2"]
subprocess.run(["python", "../main.py", "-m", args_all[0], args_all[1], args[0], args[1], args[2], args[3]])

args = ["model.kernel_layout=[1,3]", "model.depth=40", "model.widen_factor=2"]
subprocess.run(["python", "../main.py", "-m", args_all[0], args_all[1], args[0], args[1], args[2], args[3]])

args = ["model.kernel_layout=[3,1,1]", "model.depth=40", "model.widen_factor=2"]
subprocess.run(["python", "../main.py", "-m", args_all[0], args_all[1], args[0], args[1], args[2], args[3]])

args = ["model.kernel_layout=[3,1,3]", "model.depth=22", "model.widen_factor=2"]
subprocess.run(["python", "../main.py", "-m", args_all[0], args_all[1], args[0], args[1], args[2], args[3]])


# experiment 2 Number of convolutional layers per residual block
# ran all
# wrn-40-2-[3] params: 
args = ["model.kernel_layout=[3]", "model.depth=40", "model.widen_factor=2"]
subprocess.run(["python", "../main.py", "-m", args_all[0], args_all[1], args[0], args[1], args[2], args[3]])
# wrn-40-2-[3,3,3,3] params: 
args = ["model.kernel_layout=[3,3,3,3]", "model.depth=40", "model.widen_factor=2"]
subprocess.run(["python", "../main.py", "-m", args_all[0], args_all[1], args[0], args[1], args[2], args[3]])
# wrn-40-2-[3,3,3] params: 
args = ["model.kernel_layout=[3,3,3]", "model.depth=40", "model.widen_factor=2", "wandb.notes=exp_2_wrn"]
subprocess.run(["python", "../main.py", "-m", args_all[0], args_all[1], args[0], args[1], args[2], args[3]])


# experiment 3 Width of residual blocks
# d=40, k=1,2,4,8, [3,3]
args = ["model.kernel_layout=[3,3]", "model.depth=40", "model.widen_factor=1,2,4,8", "wandb.notes=exp_3_eq_wrn"]
subprocess.run(["python", "../main.py", "-m", args_all[0], args_all[1], args[0], args[1], args[2], args[3]])

args_all = ["model=eq_wrn", "model.restrict=invariant"]
# d=28, k=10, [3,3]
args = ["model.kernel_layout=[3,3]", "model.depth=28", "model.widen_factor=10,12", "wandb.notes=exp_3_eq_wrn"]
subprocess.run(["python", "../main.py", "-m", args_all[0], args_all[1], args[0], args[1], args[2], args[3]])
# d=16,22, k=8, [3,3]
args = ["model.kernel_layout=[3,3]", "model.depth=16,22", "model.widen_factor=8,10", "wandb.notes=exp_3_eq_wrn"]
subprocess.run(["python", "../main.py", "-m", args_all[0], args_all[1], args[0], args[1], args[2], args[3]])


# experiment 4 Dropout in residual blocks
# d=16 k=4, [3,3], drop_out=0.0,0.3
args = ["model.kernel_layout=[3,3]", "model.depth=16", "model.widen_factor=4", "model.drop_out=0.0,0.3", "wandb.notes=exp_4_eq_wrn"]
subprocess.run(["python", "../main.py", "-m", args_all[0], args_all[1], args[0], args[1], args[2], args[3], args[4]])
# d=28 k=10, [3,3], drop_out=0.0,0.3
args = ["model.kernel_layout=[3,3]", "model.depth=28", "model.widen_factor=10", "model.drop_out=0.0,0.3", "wandb.notes=exp_4_eq_wrn"]
subprocess.run(["python", "../main.py", "-m", args_all[0], args_all[1], args[0], args[1], args[2], args[3], args[4]])
# d=52 k=1, [3,3], drop_out=0.0,0.3
args = ["model.kernel_layout=[3,3]", "model.depth=52", "model.widen_factor=1", "model.drop_out=0.0,0.3", "wandb.notes=exp_4_eq_wrn"]
subprocess.run(["python", "../main.py", "-m", args_all[0], args_all[1], args[0], args[1], args[2], args[3], args[4]])
