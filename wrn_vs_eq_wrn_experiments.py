import subprocess

# python networks/eq_wrn.py -m model.kernel_layout=[3,3],[3,1],[1,3],[3,1,3],[1,3,1],[3,1,1]
# kernel_layout = [3,3],[3,1],[1,3],[3,1,3],[1,3,1],[3,1,1]

"""
# experiment 1 Type of convolutions in residual block
# fix params works for all (tested)
# already ran this experiment for wrn
args = ["model=wrn,eq_wrn", "model.restrict=invariant", "model.kernel_layout=[3,3]", "model.depth=28", "model.widen_factor=2"]
subprocess.run(["python", "experiment/main.py", "-m", args[0], args[1], args[2], args[3], args[4]])

args = ["model=wrn,eq_wrn", "model.restrict=invariant", "model.kernel_layout=[1,3,1]", "model.depth=40", "model.widen_factor=2"]
subprocess.run(["python", "experiment/main.py", "-m", args[0], args[1], args[2], args[3], args[4]])

args = ["model=wrn,eq_wrn", "model.restrict=invariant", "model.kernel_layout=[3,1]", "model.depth=40", "model.widen_factor=2"]
subprocess.run(["python", "experiment/main.py", "-m", args[0], args[1], args[2], args[3], args[4]])

args = ["model=wrn,eq_wrn", "model.restrict=invariant", "model.kernel_layout=[1,3]", "model.depth=40", "model.widen_factor=2"]
subprocess.run(["python", "experiment/main.py", "-m", args[0], args[1], args[2], args[3], args[4]])

args = ["model=wrn,eq_wrn", "model.restrict=invariant", "model.kernel_layout=[3,1,1]", "model.depth=40", "model.widen_factor=2"]
subprocess.run(["python", "experiment/main.py", "-m", args[0], args[1], args[2], args[3], args[4]])

"""
args = ["model=wrn,eq_wrn", "model.restrict=invariant", "model.kernel_layout=[3,1,3]", "model.depth=22", "model.widen_factor=2"]
subprocess.run(["python", "experiment/main.py", "-m", args[0], args[1], args[2], args[3], args[4]])


# experiment 2 Number of convolutional layers per residual block

# wrn-40-2-[3] params: 
args = ["model=wrn,eq_wrn", "model.restrict=invariant", "model.kernel_layout=[3]", "model.depth=40", "model.widen_factor=2"]
subprocess.run(["python", "experiment/main.py", "-m", args[0], args[1], args[2], args[3], args[4]])
# wrn-40-2-[3,3,3,3] params: 
args = ["model=wrn,eq_wrn", "model.restrict=invariant", "model.kernel_layout=[3,3,3,3]", "model.depth=40", "model.widen_factor=2"]
subprocess.run(["python", "experiment/main.py", "-m", args[0], args[1], args[2], args[3], args[4]])
# wrn-40-2-[3,3,3] params: 
# Start manually have to change the code
# args = ["model=wrn,eq_wrn", "model.restrict=invariant", "model.kernel_layout=[3,3,3]", "model.depth=40", "model.widen_factor=2"]
# subprocess.run(["python", "experiment/main.py", "-m", args[0], args[1], args[2], args[3], args[4]])


# experiment 3 Width of residual blocks
# d=40, k=1,2,4,8, [3,3]
args = ["model=wrn,eq_wrn", "model.restrict=invariant", "model.kernel_layout=[3,3]", "model.depth=40", "model.widen_factor=1,2,4,8"]
subprocess.run(["python", "experiment/main.py", "-m", args[0], args[1], args[2], args[3], args[4]])
# d=28, k=10, [3,3]
args = ["model=wrn,eq_wrn", "model.restrict=invariant", "model.kernel_layout=[3,3]", "model.depth=28", "model.widen_factor=10,12"]
subprocess.run(["python", "experiment/main.py", "-m", args[0], args[1], args[2], args[3], args[4]])
# d=16,22, k=8, [3,3]
args = ["model=wrn,eq_wrn", "model.restrict=invariant", "model.kernel_layout=[3,3]", "model.depth=16,22", "model.widen_factor=8,10"]
subprocess.run(["python", "experiment/main.py", "-m", args[0], args[1], args[2], args[3], args[4]])


# experiment 4 Dropout in residual blocks
# d=16 k=4, [3,3], drop_out=0.0,0.3
args = ["model=wrn,eq_wrn", "model.restrict=invariant", "model.kernel_layout=[3,3]", "model.depth=16", "model.widen_factor=4", "model.drop_out=0.0,0.3"]
subprocess.run(["python", "experiment/main.py", "-m", args[0], args[1], args[2], args[3], args[4], args[5]])
# d=28 k=10, [3,3], drop_out=0.0,0.3
args = ["model=wrn,eq_wrn", "model.restrict=invariant", "model.kernel_layout=[3,3]", "model.depth=28", "model.widen_factor=10", "model.drop_out=0.0,0.3"]
subprocess.run(["python", "experiment/main.py", "-m", args[0], args[1], args[2], args[3], args[4], args[5]])
# d=52 k=1, [3,3], drop_out=0.0,0.3
args = ["model=wrn,eq_wrn", "model.restrict=invariant", "model.kernel_layout=[3,3]", "model.depth=52", "model.widen_factor=1", "model.drop_out=0.0,0.3"]
subprocess.run(["python", "experiment/main.py", "-m", args[0], args[1], args[2], args[3], args[4], args[5]])
