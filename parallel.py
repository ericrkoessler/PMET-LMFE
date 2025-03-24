#!/scratch/ekoessle/myenvs/PyEnv1/bin/python
#SBATCH -p debug
#SBATCH --job-name=parallel
#SBATCH --nodes=1 --ntasks=1
#SBATCH --mem-per-cpu=1GB
#SBATCH -t 1:00:00
#SBATCH --output=out_par.out
#SBATCH --error=err_par.err

NARRAY = str(10) # number of parallel jobs
filename = "10ps_mol1000"

import os, sys
import subprocess
import time
import numpy as np
from pathlib import Path

manual = 0
JOBIDnum = 22951491
ARRAYJOBIDnum = 22951498

bindex = sys.argv[1]

if(manual==1):
    JOBID = str(JOBIDnum)
    ARRAYJOBID = str(ARRAYJOBIDnum)
    os.chdir("bindex" + bindex)
else:
    JOBID = str(os.environ["SLURM_JOB_ID"]) # get ID of this job

    Path("bindex" + bindex + "/tmpdir_" + JOBID).mkdir(parents=True, exist_ok=True) # make temporary directory for individual job files
    os.chdir("bindex" + bindex + "/tmpdir_" + JOBID) # change to temporary directory
    command = str("sbatch -W --array [1-" + NARRAY + "] ../../sbatcharray.py") # command to submit job array

    open(filename,'a').close()

    t0 = time.time()

    ARRAYJOBID = str(subprocess.check_output(command, shell=True)).replace("b'Submitted batch job ","").replace("\\n'","") # runs job array and saves job ID of the array

    t1 = time.time()
    print("Job ID: ", JOBID, flush=True)
    print("Array time: ", t1-t0, flush=True)

    os.chdir("..") # go back to original directory
    
for i in range(int(NARRAY)):
    if(i==0):
        popsum = np.loadtxt("tmpdir_" + JOBID + "/popsum_" + ARRAYJOBID + "_" + str(1) + ".txt") # load first job
    else:
        popsum += np.loadtxt("tmpdir_" + JOBID + "/popsum_" + ARRAYJOBID + "_" + str(i+1) + ".txt") # add each job together
        
popsum = popsum / int(NARRAY) # divide to go from sum to average
popsumFile = np.savetxt(f"./popsum_{filename}.txt",popsum)

t1 = time.time()
print("Total time: ", t1-t0, flush=True)

#os.system("rm -rf tmpdir_" + JOBID) # delete temporary folder (risky since this removes original job file data)