filename_top = "Run_PMET" # choose a name for this simulation run

method_filename = "LMFE"
model_filename = "model_PMET"
helperFunctions_filename = "helper_functions"
LFD_filename = "LossFuncData"

import os, sys
import shutil
from pathlib import Path
sys.path.append(os.getcwd()) 

dirname = filename_top
while(os.path.exists(Path(dirname))):
    if(dirname==filename_top):
        dirname = filename_top + "_2"
    else:
        dirind = str(int(dirname.split("_")[-1]) + 1)
        dirname = filename_top + "_" + dirind
Path(dirname).mkdir(parents=True, exist_ok=True) # make directory for all files to be copied in

shutil.copyfile("Run.py", dirname + "/Run.py")
shutil.copyfile(method_filename + ".py", dirname + "/" + method_filename + ".py")
shutil.copyfile(model_filename + ".py", dirname + "/" + model_filename + ".py")
shutil.copyfile(helperFunctions_filename + ".py", dirname + "/" + helperFunctions_filename + ".py")
shutil.copyfile(LFD_filename + ".txt", dirname + "/" + LFD_filename + ".txt")

os.chdir(dirname)

Path("data").mkdir(parents=True, exist_ok=True)
shutil.copyfile(model_filename + ".py", "data/model.py")
shutil.copyfile(method_filename + ".py", "data/method.py")
shutil.copyfile(helperFunctions_filename + ".py", "data/helper_functions.py")
shutil.copyfile(LFD_filename + ".txt", "data/LossFuncData.txt")
    
os.chdir("data")
sys.path.append(os.getcwd()) 

import method
import model as m

result = method.runTraj()

popsumFile = open(f"./popsum_{filename_top}.txt","w")

for t in range(result.shape[-1]):
    popsumFile.write(f"{t * m.NSkip * m.dtN / m.ps_au} \t")
    for i in range(4):
        popsumFile.write(str(result[i,t].real / m.NTraj) + "\t")
    popsumFile.write("\n")
    
popsumFile.close()