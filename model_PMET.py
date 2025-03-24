import numpy as np

### Conversion constants ###

eV_au = 0.0367493036 # eV to au
meV_au = 0.0000367493036 # meV to au
ps_au = 41341.374575751 # picosecond to au
fs_au = 41.341374575751 # femtosecond to au

### Parameters ###

NSteps = 80000 # number of nuclear steps in the simulation
NStepsPrint = 500 # number of nuclear steps that are stored/printed/outputed
NSkip = int(NSteps/NStepsPrint) # used to enforce NStepsPrint
totalTime = 10 * ps_au # total amount of simulation time
dtN = totalTime/NSteps # nuclear timestep 
EStep = 2 # number of electronic timesteps per half-nuclear step
dtE = dtN/EStep # electronic timestep
NTraj = 1 # number of trajectories to compute (sequentially)
NMol = 1000 # number of molecules (one donor state and one acceptor state per molecule)
NStates = 2 + 2*NMol # total number of states
NR = NMol # total number of nuclear DOFs
lamD = 0 * meV_au # reorganization energy of the donor (number is the energy in meV)
lamA = 100 * meV_au # reorganization energy of the acceptor
zeta  = 25.789 * ps_au # friction constant
ws = 14.498 / ps_au # solvent frequency
ms = 0.265 * ps_au**2 # solvent mass
RD0 = np.sqrt(2*lamD/(ms*ws**2)) # shift of donor PES minimum
RA0 = np.sqrt(2*lamA/(ms*ws**2)) # shift of acceptor PES minimum
dHD = -ms * ws**2 * RD0 # linear coupling strength for donor state
dHA = -ms * ws**2 * RA0 # linear coupling strength for acceptor state
VDA = 10 * meV_au # diabatic coupling between the donor and acceptor
beta = 1053 # 1/kT of bath in au (1053 is 300K)
G = 300 * meV_au + lamD # adjusted driving force outside the cavity
ED = 3 * eV_au - lamD # energy of donor
EA = ED + G # energy of acceptor
wc = 3 * eV_au # photon energy
gc = 300 * meV_au/np.sqrt(NMol) # light-matter coupling (number is total collective coupling in meV)
Phi = 0.5*np.arctan2(2*np.sqrt(NMol)*gc,wc-ED-lamD) # mixing angle
gamG1toG0 = 0 * meV_au # cavity loss rate
VL = 10 * meV_au # laser coupling strength
wL = 0.5*(ED+lamD+wc)+0.5*np.sqrt((2*gc*np.sqrt(NMol))**2+(ED+lamD-wc)**2) # laser driving frequency tuned to UP
wLR = np.sqrt(4*VL**2 + (wL-wc)**2) # laser Rabi frequency
lang1 = 2*dtN/(2*ms+dtN*zeta) # Langevin constant
lang2 = np.sqrt(dtN*zeta/(2*beta)) # Langevin constant
lang3 = lang1*zeta # Langevin constant
lang4 = lang2*2*ms/(dtN*zeta) # Langevin constant
initState = 'G0' # choice of G0, D0, UP, LP
