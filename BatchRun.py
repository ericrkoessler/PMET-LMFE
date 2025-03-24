#!/scratch/ekoessle/myenvs/PyEnv1/bin/python
#SBATCH -p debug
#SBATCH --job-name=batchrun
#SBATCH --nodes=1 --ntasks=1
#SBATCH --mem-per-cpu=1GB
#SBATCH -t 1:00:00
#SBATCH --output=out_batch.out
#SBATCH --error=err_batch.err

filename_top = "BatchRun_PMET"
parallel_filename = "parallel"
sbatcharray_filename = "sbatcharray"
method_filename = "LMFE"
model_filename = "model_PMET"
input_filename = "input_batch"
helperFunctions_filename = "helper_functions"
LFD_filename = "LossFuncData"

import os, sys
import shutil
import importlib
import subprocess
import numpy as np
from pathlib import Path
sys.path.append(os.getcwd()) 
inp = importlib.import_module(input_filename)

dirname = filename_top
while(os.path.exists(Path(dirname))):
    if(dirname==filename_top):
        dirname = filename_top + "_2"
    else:
        dirind = str(int(dirname.split("_")[-1]) + 1)
        dirname = filename_top + "_" + dirind
Path(dirname).mkdir(parents=True, exist_ok=True) # make directory for all files to be copied in

shutil.copyfile("BatchRun.py", dirname + "/BatchRun.py")
#shutil.copyfile("BatchInput.py", dirname + "/BatchInput.py")
shutil.copyfile(parallel_filename + ".py", dirname + "/" + parallel_filename + ".py")
shutil.copyfile(sbatcharray_filename + ".py", dirname + "/" + sbatcharray_filename + ".py")
shutil.copyfile(method_filename + ".py", dirname + "/" + method_filename + ".py")
shutil.copyfile(model_filename + ".py", dirname + "/" + model_filename + ".py")
shutil.copyfile(input_filename + ".py", dirname + "/" + input_filename + ".py")
shutil.copyfile(helperFunctions_filename + ".py", dirname + "/" + helperFunctions_filename + ".py")
shutil.copyfile(LFD_filename + ".txt", dirname + "/" + LFD_filename + ".txt")

os.chdir(dirname)

model_file_data = open(model_filename + ".py", 'r').readlines()
update_lines = np.zeros(len(inp.var_dict), dtype=int)
for i in range(len(inp.var_dict)):
    for num, line in enumerate(model_file_data):
        if(line.startswith(list(inp.var_dict)[i] + " =") or line.startswith(list(inp.var_dict)[i] + "=")):
            update_lines[i] = num

for i in range(inp.batch_size):
    Path("bindex" + str(i+1)).mkdir(parents=True, exist_ok=True)
    model_update_file = open("bindex" + str(i+1) + "/model_update.py","w")
    model_update_file.write("# This is a vanity file to show parameters specific to this directory\n\n")
    model_update_file.write("import model\n\n")
    for j in range(len(inp.var_dict)):
        model_update_file.write("model." + list(inp.var_dict)[j] + " = " + repr(list(inp.var_dict.values())[j][i]))
        model_update_file.write("\n")
    model_update_file.close()
    for j in range(len(inp.var_dict)):
        model_file_data[update_lines[j]] = list(inp.var_dict)[j] + " = " + repr(list(inp.var_dict.values())[j][i]) + "\n"
    model_file_tmp = open("bindex" + str(i+1) + "/model.py", 'w')
    model_file_tmp.writelines(model_file_data)
    model_file_tmp.close()
    shutil.copyfile(method_filename + ".py", "bindex" + str(i+1) + "/method.py")
    shutil.copyfile(helperFunctions_filename + ".py", "bindex" + str(i+1) + "/helper_functions.py")
    shutil.copyfile(LFD_filename + ".txt", "bindex" + str(i+1) + "/LossFuncData.txt")
    
for i in range(inp.batch_size):
    command = str("sbatch " + parallel_filename + ".py " + str(i+1)) # command to submit parallel file
    subprocess.Popen(command, shell=True)
