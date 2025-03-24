import numba as nb
import numpy as np
from numpy import random as rn
import time
import helper_functions as hf
import model as m

@nb.jit(nopython=True)
def initRP(sd): # initialization of wavefunction coefficients
    σP = np.sqrt(m.ms*m.ws/(2.0 * np.tanh(0.5 * m.beta * m.ws))) # standard dev of nuclear momentum (Wigner distribution)
    σR = σP/(m.ms*m.ws) # standard dev of nuclear position
    for tr in range(m.NTraj):
        for j in range(m.NMol):
            sd.R[j, tr] = rn.normal() * σR # gaussian random variable centered around 0
            sd.P[j, tr] = rn.normal() * σP # gaussian random variable centered around 0

@nb.jit(nopython=True)
def initc(sd): # initialization of wavefunction coefficients
    if(m.initState=='G0'):
        sd.c[0, :] = 1.0*np.ones(m.NTraj)
    elif(m.initState=='D0'):
        sd.c[1, :] = 1.0*np.ones(m.NTraj)
    elif(m.initState=='UP'):
        sd.c[1:1+m.NMol, :] = np.sin(m.Phi)/np.sqrt(m.NMol)*np.ones((m.NMol,m.NTraj))
        sd.c[1+2*m.NMol, :] = np.cos(m.Phi)*np.ones(m.NTraj)
    elif(m.initState=='LP'):
        sd.c[1:1+m.NMol, :] = np.cos(m.Phi)/np.sqrt(m.NMol)*np.ones((m.NMol,m.NTraj))
        sd.c[1+2*m.NMol, :] = -np.sin(m.Phi)*np.ones(m.NTraj)
        
@nb.jit(nopython=True)
def LBLoss(sd, dt, gam, States, LFD):
    # decay States[0] to States[1]

    for tr in range(m.NTraj):
        c0 = sd.c[States[0],tr]
        c1 = sd.c[States[1],tr]
        
        # Total population of both states
        pop0 = np.abs(c0*np.conj(c0))
        pop1 = np.abs(c1*np.conj(c1))
        
        # Propogated coefficients without random phase
        c0P = np.exp(-gam*dt/2)*c0
        c1P = np.sqrt(pop1+pop0*(1-np.exp(-gam*dt)))
        
        if(c1P>0):
            # Determine off-diagonal decay rate
            dr = np.abs(c1)/c1P
            
            if(dr<1.0):
                drnum = dr*100000
                dTheta = LFD[int(np.floor(drnum)),1]*(1+np.floor(drnum)-drnum) + LFD[int(np.floor(drnum))+1,1]*(drnum-np.floor(drnum))
            else:
                dTheta = 0.0
            
            # Calculate Theta from uniform distribution [-dTheta,dTheta]
            Theta = rn.uniform(-dTheta,dTheta)
            
            # Propogate random phase of coefficients
            c1P = np.exp(1j*(Theta+np.angle(c1)))*c1P

        sd.c[States[0],tr] = c0P
        sd.c[States[1],tr] = c1P
        
@nb.jit(nopython=True)
def updatec(sd, t, dt): # this updates the quantum state
    sd.c_3[:, :] = sd.c[:, :]
    hf.mreal(sd.c[:, :])
    hf.mimag(sd.c_3[:, :])
     
    hf.Hxc(sd.c, sd) # changes sd.c_1
    hf.mpm(sd.c_3, 1.0, sd.c_1, -0.5 * dt, sd.c_3, 0.0)

    hf.Hxc(sd.c_3, sd) # changes sd.c_1
    hf.mpm(sd.c, 1.0, sd.c_1, dt, sd.c, 0.0)

    hf.Hxc(sd.c, sd) # changes sd.c_1
    hf.mpm(sd.c_3, 1.0, sd.c_1, -0.5 * dt, sd.c_3, 0.0)
    
    # Add real and imag back together
    hf.mpm(sd.c, 1.0, sd.c_3, 1.0j, sd.c, 0.0)
    
    hf.LaserPropG0G1(sd, t, dt)
    
@nb.jit(nopython=True)
def updateR(sd, dt, η):
    sd.F_2[:,:] = 0.0
    hf.mpm(sd.P, 1.0, sd.F, dt/2, sd.F_2, 0.0)
    hf.mpm(sd.F_2, 1.0, η, m.lang2, sd.F_2, 0.0)
    hf.mpm(sd.F_2, m.lang1, sd.R, 1.0, sd.R, 0.0)
    
@nb.jit(nopython=True)
def updateP(sd, dt, η):
    sd.F_2[:,:] = 0.0
    hf.mpm(sd.P, -1.0, sd.F, -dt/2, sd.F_2, 0.0)
    hf.mpm(sd.F_2, 1.0, η, m.lang4, sd.F_2, 0.0)
    hf.mpm(sd.F_2, m.lang3, sd.P, 1.0, sd.P, 0.0)
    hf.mpmp(sd.F, dt/2, sd.F_1, dt/2, sd.P, 0.0)
    
@nb.jit(nopython=True)
def updateF(sd): # this calculates the classical force on each nuclear DOF using a mean-field approach (hence Mean-Field Ehrenfest)
    sd.F_1[:,:] = 0.0
    hf.mabs2(sd.c, sd.c_4)
    for j in range(m.NMol):
        sd.F_3[:] = 0.0
        hf.vpv(sd.R[j, :], -m.ms*m.ws**2, sd.F_1[j, :], 0.0, sd.F_1[j, :], 0.0) # state-independent force
        hf.vpv(sd.c_4[1+j,:], -m.dHD, sd.F_3, 1.0, sd.F_3, 0.0) # donor force
        hf.vpv(sd.c_4[1+m.NMol+j,:], -m.dHA, sd.F_3, 1.0, sd.F_3, 0.0) # acceptor force
        hf.vpvp(sd.F_1[j, :], 0.0, sd.F_3, 1.0, sd.F_1[j, :], 0.0)
        
@nb.jit(nopython=True)
def Propagate(sd, LFD, ti): # this updates the nuclear DOFs using Langevin Velocity Verlet (quantum state is also updated within)

    LBLoss(sd, m.dtN/4, m.gamG1toG0, [1+2*m.NMol,0], LFD) # cavity loss
    for t in range(m.EStep): # do EStep number of electronic steps, using a timestep of dtE/2
        updatec(sd, ti + t * m.dtE/2, m.dtE/2)
    LBLoss(sd, m.dtN/4, m.gamG1toG0, [1+2*m.NMol,0], LFD) # cavity loss
    
    η = rn.normal(0.0, 1.0, sd.R.shape)
    updateR(sd, m.dtN, η)
                
    LBLoss(sd, m.dtN/4, m.gamG1toG0, [1+2*m.NMol,0], LFD) # cavity loss
    hf.HRupdate(sd)
    for t in range(m.EStep): # do EStep number of electronic steps, using a timestep of dtE/2
        updatec(sd, ti + m.dtN/2 + t * m.dtE/2, m.dtE/2)
    LBLoss(sd, m.dtN/4, m.gamG1toG0, [1+2*m.NMol,0], LFD) # cavity loss
    
    updateF(sd)
    updateP(sd, m.dtN, η)
    
    sd.F[:, :] = sd.F_1[:, :]

def runTraj(): # this is the main function that runs everything; call this function to run this method
    
    t0 = time.time()
    
    LFD = np.loadtxt("../LossFuncData.txt")

    popsum = np.zeros((4,m.NStepsPrint))
    
    sd = hf.state_data()
    
    initRP(sd)
    
    initc(sd)
    
    hf.HRupdate(sd)
    updateF(sd)
    sd.F[:, :] = sd.F_1[:, :]
    
    t1 = time.time()
    print("Init time: ", t1-t0, flush=True)
        
    iskip = 0 # counting variable to determine when to store the current timestep data
    for i in range(m.NSteps):
        #------- ESTIMATORS-------------------------------------
        if (i % m.NSkip == 0): # this is what lets NSkip choose which timesteps to store
            for tr in range(m.NTraj):
                pop = np.abs(sd.c[:,tr])**2
                popsum[0,iskip] += pop[0] # G0 pop
                popsum[1,iskip] += np.sum(pop[1:1+m.NMol]) # D0 pop
                popsum[2,iskip] += np.sum(pop[1+m.NMol:1+2*m.NMol]) # A0 pop
                popsum[3,iskip] += pop[1+2*m.NMol] # G1 pop
            iskip += 1
            print("Step " + str(iskip) + "/" + str(m.NStepsPrint) + " running", flush=True)
        #-------------------------------------------------------
        Propagate(sd, LFD, i*m.dtN) # update variables for next nuclear timestep
        
    t2 = time.time()
    print("Propagation time: ", t2-t1, flush=True)

    return popsum # runTraj() returns these variables as output