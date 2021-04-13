#Purpose: Takes in wavefuction u and a part of the Hamiltonian. Takes in not-distributed arrays The wavefunction should be in the right basis (position or momentum). Return the energy of the wavefuction u, based on a part of the Hamiltonian.
#Returns: The energy contribution from part of the Hamiltonian.
#Author : Andrew Cleary
#Date   : 24/06/2020

import matplotlib.pylab as plt
import numpy as np

#--------------------basic variables------------------

"""Physical Parameters"""
#u...............wavefunction
#h...............part of hamiltonian

"""Numeric Parameters"""

#-------------------function--------------------------

def energy(u,h):

#--------------create hermitian conjugate-------------

    udag = np.conj(u.T)
    
#----------------multiply everything------------------

    u = np.multiply(h,u)
    u = np.multiply(udag,u)
    
#-------------------integrate------------------------
    
    return np.sum(u)
