#!/scratch/ekoessle/myenvs/PyEnv1/bin/python
#SBATCH -p debug
#SBATCH --job-name=sbatcharray
#SBATCH --nodes=1 --ntasks=1
#SBATCH --mem-per-cpu=1GB
#SBATCH -t 1:00:00
#SBATCH --output=out_%A_%a.out
#SBATCH --error=err_%A_%a.err

import sys, os
sys.path.append(os.popen("pwd").read().split("/tmpdir")[0]) # include parent directory which has method and model files
#-------------------------
# import LMFELANG_stack_1mode as method
# import PMETmodelLANG_stack_1mode as model
import method
import model as m
import time
import numpy as np

JOBID = str(os.environ["SLURM_ARRAY_JOB_ID"]) # get ID of this job
TASKID = str(os.environ["SLURM_ARRAY_TASK_ID"]) # get ID of this task within the array

if(int(TASKID)==1):
    open('ARRAYJOBID_' + JOBID,'a').close()

t0 = time.time()

result = method.runTraj() # run the method

t1 = time.time()
print("Job time: ",t1-t0, flush=True)

# Write individual job data to files

popsumFile = open(f"./popsum_{JOBID}_{TASKID}.txt","w")

for t in range(result.shape[-1]):
    popsumFile.write(f"{t * m.NSkip * m.dtN / m.ps_au} \t")
    for i in range(4):
        popsumFile.write(str(result[i,t].real / m.NTraj) + "\t")
    popsumFile.write("\n")
    
popsumFile.close()

t2 = time.time()
print("Total time: ", t2-t0, flush=True)