import subprocess
import re
import time

def check_gpu_usage():
    cmd = 'nvidia-smi --query-gpu=utilization.gpu --format=csv'
    output = subprocess.check_output(cmd.split())
    gpu_usage = int(re.findall('\d+', output.decode().strip().split('\n')[1])[0])
    print('GPU usage: {}%'.format(gpu_usage))
    return gpu_usage

def main():
    gpu_usage = check_gpu_usage()
    if gpu_usage < 5:
        print('GPU usage low check again in 10 seconds')
        time.sleep(10)
        gpu_usage = check_gpu_usage()
        if gpu_usage < 5:
            print('GPU usage still low, running experiment')
            subprocess.run(["python", "experiment/run_files/efficientnet_experiments.py"])


if __name__ == "__main__":
    main()
