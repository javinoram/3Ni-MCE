import numpy as np
import numpy.linalg as la
import pandas as pd
import sys

"""
Seccion de funciones de observables.
Aca se definen ciertas funciones utilies para calcular observables.
Estas formulas fueron extraidas de https://www.mdpi.com/1996-1944/13/19/4399

ee: vector de valores propios (meV)
t: temperatura (K)
partition: vector de exp(-E_i / T*k_b) para cada valor propio
mag: vector de valores esperados de la magnetizacion para cada vector propio
rho: operador de densidad
spin: numero de espines del sistema
spin_val: tamaño del vector de estado de cada sitio individual (spin 0.5 -> 2, spin 1 -> 3)
"""

dtype = 'float128'
boltz = 8.617333262e-2 #mev/K
gyro = 2.0
nub = 5.7883818066e-2 #meV/T

def NumpySpecific_heat(ee, t):
    partition = np.exp( np.divide(-ee, t*boltz, dtype=dtype), dtype=dtype )
    Z = np.sum(partition, dtype=dtype)
    partition = np.divide( partition, Z, dtype=dtype )

    aux1 = np.sum(partition*ee, dtype=dtype)
    aux2 = np.sum(partition*(ee**2), dtype=dtype)
    return np.divide(aux2- (aux1**2), (t*t*boltz), dtype=dtype)


def NumpyEntropy(ee, t):
    partition = np.exp( np.divide(-ee, t*boltz, dtype=dtype), dtype=dtype )
    Z = np.sum(partition, dtype=dtype)
    partition = np.divide( partition, Z, dtype=dtype )

    termal = np.sum( ee*partition, dtype=dtype)
    free_energy = -boltz*np.log( Z, dtype=dtype) 
    return np.divide(termal, t, dtype=dtype) - free_energy 

def NumpyvonNeumann(ee):
    aux = np.sum( -ee*np.log(ee, dtype=dtype), dtype=dtype )
    return aux

def NumpyPartialTraceLR(rho, spin, spin_val):
    aux = np.zeros((int(rho.shape[0]/spin_val), int(rho.shape[1]/spin_val)))
    for i in range(spin_val):
        valaux = spin_val**(spin-1)
        aux = aux + rho[valaux*i:valaux*(i+1), valaux*i:valaux*(i+1)]
    return aux

def NumpyPartialTraceRL(rho, spin, spin_val):
    aux = np.zeros((int(rho.shape[0]/spin_val), int(rho.shape[1]/spin_val)))
    for i in range(spin_val**(spin-1)):
        for j in range(spin_val**(spin-1)):
            aux[i,j] = np.trace( rho[spin_val*i:spin_val*(i+1), spin_val*j:spin_val*(j+1)] )
    return aux

def NumpyMagnetization(mag, ee, t):
    partition = np.exp( np.divide(-ee, t*boltz, dtype=dtype), dtype=dtype )
    Z = np.sum(partition, dtype=dtype)
    partition = np.divide( partition, Z, dtype=dtype )
    mag = np.sum( mag*partition, dtype=dtype)
    return mag

def Numpyget_eigen(H):
    ee, vv= np.linalg.eigh(H)
    return ee, vv

"""
Seccion de parametros del hamiltoniano, aca se define la funcion que construye el hamiltoniano.
Ademas de sus elementos claves.

params: lista de los parametros del hamiltoniano definido
Considerar que cambios aqui implican cambiar los scripts para ya que estos estan implementeados para 
4 parametros. Pero la estructura general deberia mantenerse.
"""
def hamiltoniano(params):
    #J1, J2, J13, h_z, h_x
    H = -2*params[0]*Int1 -2*params[1]*Int2 -2*params[2]*Int3 -gyro*nub*params[3]*OZ -gyro*nub*params[4]*OX
    H = np.real(H)
    return H


sI = np.array( [ [1,0,0], [0,1,0], [0,0,1] ] ,dtype='float64')
sX = (1.0/np.sqrt(2))*np.array( [ [0,1,0], [1,0,1], [0,1,0] ], dtype='float64') 
sY = (1.0/np.sqrt(2))*np.array( [ [0,-1j,0], [1j,0,-1j], [0,1j,0] ], dtype='complex64') 
sZ = np.array( [ [1,0,0], [0,0,0], [0,0,-1] ], dtype='float64') 


Int1 = np.kron(sZ, np.kron(sZ, sI))  +\
    np.kron(sX, np.kron(sX, sI)) +\
    np.kron(sY, np.kron(sY, sI)) 

Int2 = np.kron(sI, np.kron(sZ, sZ))  +\
    np.kron(sI, np.kron(sX, sX)) +\
    np.kron(sI, np.kron(sY, sY)) 

Int3 =  np.kron(sZ, np.kron(sI, sZ))  +\
    np.kron(sX, np.kron(sI, sX)) +\
    np.kron(sY, np.kron(sI, sY)) 

OZ = np.kron(sZ, np.kron(sI, sI))  +\
    np.kron(sI, np.kron(sZ, sI)) +\
    np.kron(sI, np.kron(sI, sZ)) 

OX = np.kron(sX, np.kron(sI, sI))  +\
    np.kron(sI, np.kron(sX, sI)) +\
    np.kron(sI, np.kron(sI, sX)) 

boltz = 8.617333262e-2 #mev/K
gyro = 2.0
nub = 5.7883818066e-2 #meV/T

J3D = [1.49, 1.49, -0.89]
J1D = [-0.08, -0.08, 0.0]
