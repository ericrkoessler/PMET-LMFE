import numpy as np

batch_size = 11

eV_au = 0.0367493036 # eV to au
gamG1toG0 = np.array([3*eV_au/10, 3*eV_au/20, 3*eV_au/50, 3*eV_au/100, 3*eV_au/200, 3*eV_au/500, 3*eV_au/1000, 3*eV_au/2000, 3*eV_au/5000, 3*eV_au/10000, 0])

var_dict = {'gamG1toG0': gamG1toG0}