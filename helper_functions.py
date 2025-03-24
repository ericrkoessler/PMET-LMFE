import numba as nb
import numpy as np
from numba.experimental import jitclass
import model as m

### Numba data types ###

state_types = [
    
    ### State data ###
    
    ('c', nb.complex128[:,::1]), # state wavefunction
    ('R', nb.float64[:,::1]), # nuclear positions
    ('P', nb.float64[:,::1]), # nuclear momenta
    ('F', nb.float64[:,::1]), # forces
    ('Eii', nb.float64[:,::1]), # diagonal diabatic energies
    
    ### IGNORE: initialized arrays to help speed up code ###
    
    ('c_1', nb.complex128[:,::1]), # state wavefunction copy (used to speed up code)
    ('c_2', nb.complex128[:,::1]), # state wavefunction copy (used to speed up code)
    ('c_3', nb.complex128[:,::1]), # state wavefunction copy (used to speed up code)
    ('c_4', nb.float64[:,::1]), # state wavefunction copy (used to speed up code)
    ('F_1', nb.float64[:,::1]), # forces copy (used to speed up code)
    ('F_2', nb.float64[:,::1]), # forces copy (used to speed up code)
    ('F_3', nb.float64[::1]) # forces copy (used to speed up code)
]

### Data class for storing trajectory state information ###

@jitclass(state_types)
class state_data(object): # storage object for state variables (wavefunction, nuclei, etc.)
    def __init__(self):
        
        ### State data ###
        
        self.c = np.zeros((m.NStates,m.NTraj), dtype=np.complex128) # state wavefunction
        self.R = np.zeros((m.NR,m.NTraj)) # nuclear positions
        self.P = np.zeros((m.NR,m.NTraj)) # nuclear momenta
        self.F = np.zeros((m.NR,m.NTraj)) # forces
        self.Eii = np.zeros((m.NStates,m.NTraj)) # diagonal diabatic energies
        
        ### IGNORE: initialized arrays to help speed up code ###
        
        self.c_1 = np.zeros((m.NStates,m.NTraj), dtype=np.complex128) # state wavefunction copy (used to speed up code)
        self.c_2 = np.zeros((m.NStates,m.NTraj), dtype=np.complex128) # state wavefunction copy (used to speed up code)
        self.c_3 = np.zeros((m.NStates,m.NTraj), dtype=np.complex128) # state wavefunction copy (used to speed up code)
        self.c_4 = np.zeros((m.NStates,m.NTraj)) # state wavefunction copy (used to speed up code)
        self.F_1 = np.zeros((m.NR,m.NTraj)) # forces copy (used to speed up code)
        self.F_2 = np.zeros((m.NR,m.NTraj)) # forces copy (used to speed up code)
        self.F_3 = np.zeros((m.NTraj)) # forces copy (used to speed up code)
        
### General linear algebra functions ###
        
@nb.jit(nopython=True)
def mreal(m1):
    shp1, shp2 = m1.shape
    for i in range(shp1):
        for j in range(shp2):
            m1[i,j] = m1[i,j].real
        
@nb.jit(nopython=True)
def mimag(m1):
    shp1, shp2 = m1.shape
    for i in range(shp1):
        for j in range(shp2):
            m1[i,j] = m1[i,j].imag
            
@nb.jit(nopython=True)
def mabs2(m1, m2):
    shp1, shp2 = m1.shape
    for i in range(shp1):
        for j in range(shp2):
            m2[i,j] = m1[i,j].real**2 + m1[i,j].imag**2
      
@nb.jit(nopython=True)
def vpv(v1, c1, v2, c2, v3, c3):
    shp = v1.shape[0]
    for i in range(shp):
        v3[i] = v1[i]*c1 + v2[i]*c2 + c3
        

@nb.jit(nopython=True)
def vpvp(v1, c1, v2, c2, v3, c3):
    shp = v1.shape[0]
    for i in range(shp):
        v3[i] += v1[i]*c1 + v2[i]*c2 + c3 

@nb.jit(nopython=True)
def mxmp(m1, m2, m3):
    shp1, shp2 = m1.shape
    for i in range(shp1):
        for j in range(shp2):
            m3[i, j] += m1[i, j] * m2[i, j]
            
@nb.jit(nopython=True)
def mpm(m1, c1, m2, c2, m3, c3):
    shp1, shp2 = m1.shape
    for i in range(shp1):
        for j in range(shp2):
            m3[i, j] = m1[i, j]*c1 + m2[i, j]*c2 + c3 
            
@nb.jit(nopython=True)
def mpv(m1, c1, v2, c2, m3, c3):
    shp1, shp2 = m1.shape
    for i in range(shp1):
        for j in range(shp2):
            m3[i, j] = m1[i, j]*c1 + v2[j]*c2 + c3 
            

@nb.jit(nopython=True)
def mpmp(m1, c1, m2, c2, m3, c3):
    shp1, shp2 = m1.shape
    for i in range(shp1):
        for j in range(shp2):
            m3[i, j] += m1[i, j]*c1 + m2[i, j]*c2 + c3
            
@nb.jit(nopython=True)
def cxgc(c, c_1):
    shpj, shpt = c.shape
    for t in range(shpt):
        jsum = 0.0
        for j in range(shpj):
            jsum += c[j, t]
        c_1[t] = jsum
        
@nb.jit(nopython=True)
def cxvda(c, c_1):
    shpj, shpt = c.shape
    shpj_2 = int(shpj//2)
    for t in range(shpt):
        for j in range(shpj_2):
            c_1[j, t] = c[shpj_2+j,t]
            c_1[shpj_2+j, t] = c[j,t]
            
### Method/model specific functions ###

@nb.jit(nopython=True)
def LaserPropG0G1(sd, t, dt):
    for tr in range(m.NTraj):
        sd.c_1[0,tr] = np.exp(-1j*dt*(-m.wc+m.wLR-m.wL)/2)*(-2*(sd.c[1+2*m.NMol,tr])*np.exp(1j*(m.wL*t))*(np.exp(1j*m.wLR*dt)-1)*m.VL+sd.c[0,tr]*(-m.wc+m.wLR+m.wL+np.exp(1j*m.wLR*dt)*(m.wc+m.wLR-m.wL)))/(2*m.wLR)
        sd.c_1[1+2*m.NMol,tr] = np.exp(-1j*(dt*(-m.wc+m.wLR+m.wL)+2*(m.wL*t))/2)*(-2*sd.c[0,tr]*(np.exp(1j*m.wLR*dt)-1)*m.VL+(sd.c[1+2*m.NMol,tr])*np.exp(1j*(m.wL*t))*(m.wc+m.wLR-m.wL+np.exp(1j*m.wLR*dt)*(-m.wc+m.wLR+m.wL)))/(2*m.wLR)
    sd.c[0,:] = sd.c_1[0,:]
    sd.c[1+2*m.NMol,:] = sd.c_1[1+2*m.NMol,:]
    
@nb.jit(nopython=True)
def HRupdate(sd): # updates Hamiltonian diagonal energies based on R
    sd.Eii[0, :] = 0.0 # G0 energy; energy phase rotation of -m.wc*dt already accounted for in LaserPropG0G1
    sd.Eii[1+2*m.NMol, :] = 0.0 # G1 energy; set to 0 for convinience
    for j in range(m.NMol):
        vpv(sd.R[j, :], m.dHD, sd.Eii[1+j, :], 0.0, sd.Eii[1+j, :], m.ED+m.lamD-m.wc) # D0 energy
        vpv(sd.R[j, :], m.dHA, sd.Eii[1+m.NMol+j, :], 0.0, sd.Eii[1+m.NMol+j, :], m.EA+m.lamA-m.wc) # A0 energy

@nb.jit(nopython=True)
def Hxc(c, sd): # Hamiltonian operating on a state wavefunction
    sd.c_1[:, :] = 0.0
    sd.c_2[:, :] = 0.0
    
    # Light-matter coupling
    mpv(sd.c_1[1:1+m.NMol, :], 1.0, c[1+2*m.NMol, :], m.gc, sd.c_1[1:1+m.NMol, :], 0.0)
    cxgc(c[1:1+m.NMol, :], sd.c_2[1+2*m.NMol, :])
    vpv(sd.c_2[1+2*m.NMol, :], m.gc, sd.c_1[1+2*m.NMol, :], 1.0, sd.c_1[1+2*m.NMol, :], 0.0)
    
    # Donor-acceptor coupling
    cxvda(c[1:1+2*m.NMol, :], sd.c_2[1:1+2*m.NMol, :])
    mpm(sd.c_2[1:1+2*m.NMol, :], m.VDA, sd.c_1[1:1+2*m.NMol, :], 1.0, sd.c_1[1:1+2*m.NMol, :], 0.0)

    # Diagonal energies
    mxmp(c, sd.Eii, sd.c_1)