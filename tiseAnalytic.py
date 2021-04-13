#Purpose: To return the analytic solution to the time independent Schroedinger equation, based on the parameters defined below. Generates all states up to some excitation state.
#Returns: The density of the 2d solution to the TISE.
#Author : Andrew Cleary
#Date   : 24/06/2020

import numpy as np
from scipy.constants import pi

#--------------------basic parameters------------------

"""Physical Parameters"""
#m ..............mass of particle
#hbar............ensure same hbar def
#w...............frequency of potential = 1/2*m*w^2*x^2
#state...........excitation number

"""Numeric Parameters"""
#N...............resolution of grid
#xMax............maximum x value
#yMax............maximum y value

def main(m,hbar,w,N,xMax,yMax,state=0):

    #-----------------coefficients list------------------

    Ncoeff = state+1                  #number of coefficients
    coeffs = np.zeros(Ncoeff)

    for n in range(Ncoeff):
        coeffs[n] = 1.0/np.sqrt(np.math.factorial(n)*2**n) * (m*w/pi/hbar)**(0.25)
        
    #----------------Hermite Polynomials-----------------

    Xresults = np.zeros((Ncoeff,N))
    Yresults = np.zeros((Ncoeff,N))

    x = np.zeros(N)
    y = np.zeros(N)

    dx = 2*xMax/N
    dy = 2*yMax/N

    for i in range(int(N/2)):
        x[i] = -xMax + (i+1)*dx
        x[i + int(N/2)] = (i+1)*dx

        y[i] = -yMax + (i+1)*dy
        y[i + int(N/2)] = (i+1)*dy

    for i in range(Ncoeff):
        tempCoeff = np.zeros(i+1)
        tempCoeff[i] = coeffs[i]
        Xresults[i] = coeffs[i] * np.exp(-m*w*(x**2)/2.0/hbar) * np.polynomial.hermite.hermval(x*(m*w/hbar)**(0.5),tempCoeff)
        Yresults[i] = coeffs[i] * np.exp(-m*w*(y**2)/2.0/hbar) * np.polynomial.hermite.hermval(y*(m*w/hbar)**(0.5),tempCoeff)

    X,Y = np.meshgrid(Xresults[state],Yresults[state])
    stateFull = np.multiply(X,Y)

    stateFull/=np.linalg.norm(stateFull)

    densityFull = np.square(stateFull)
    
    return densityFull

