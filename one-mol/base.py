import numpy as np
import numpy.linalg as la
import pandas as pd
import sys

"""
Constantes utiles.
    dtype: tipo de las variables
    gyro: constante giroscopica
    nub: magneton de borh
    boltz: constante de boltzman
"""
dtype = 'float128'
boltz = 8.617333262e-2 #meV/K
gyro = 2.0
nub = 5.7883818066e-2 #meV/T


"""
Funcion para calcular el calor especifico. Un detalle importante es que hay que tener cuidado a bajas temperaturas, 
debajo de 0.1 (K), ya que, puede ocurrir un overflow de la variable.
input:
    ee: vector de valores propios del hamiltoniano.
    t (K): temperatura a la que se quiere calcular el calor especifico.
output:
    valor: calor especifico a temperatura t (K).
""" 
def NumpySpecific_heat(ee, t):
    partition = np.exp( np.divide(-ee, t*boltz, dtype=dtype), dtype=dtype )
    Z = np.sum(partition, dtype=dtype)
    partition = np.divide( partition, Z, dtype=dtype )

    aux1 = np.sum(partition*ee, dtype=dtype)
    aux2 = np.sum(partition*(ee**2), dtype=dtype)
    return np.divide(aux2- (aux1**2), (t*t*boltz), dtype=dtype)


"""
Funcion para calcular la entropia. Un detalle importante es que hay que tener cuidado a bajas temperaturas, 
debajo de 0.1 (K), ya que, puede ocurrir un overflow de la variable.
input:
    ee: vector de valores propios del hamiltoniano.
    t (K): temperatura a la que se quiere calcular el calor especifico.
output:
    valor: Entropia a temperatura t (K).
""" 
def NumpyEntropy(ee, t):
    partition = np.exp( np.divide(-ee, t*boltz, dtype=dtype), dtype=dtype )
    Z = np.sum(partition, dtype=dtype)
    partition = np.divide( partition, Z, dtype=dtype )

    termal = np.sum( ee*partition, dtype=dtype)
    free_energy = -boltz*np.log( Z, dtype=dtype) 
    return np.divide(termal, t, dtype=dtype) - free_energy 


"""
Funcion para calcular la magnetizacion. Un detalle importante es que hay que tener cuidado a bajas temperaturas, 
debajo de 0.1 (K), ya que, puede ocurrir un overflow de la variable.
input:
    mag: vector de valores esperados del operador Sz respecto a cada uno de los vectores propios del hamiltoniano.
    ee: vector de valores propios del hamiltoniano.
    t (K): temperatura a la que se quiere calcular el calor especifico.
output:
    valor: Magnetizacion a temperatura t (K).
""" 
def NumpyMagnetization(mag, ee, t):
    partition = np.exp( np.divide(-ee, t*boltz, dtype=dtype), dtype=dtype )
    Z = np.sum(partition, dtype=dtype)
    partition = np.divide( partition, Z, dtype=dtype )
    mag = np.sum( mag*partition, dtype=dtype)
    return mag


"""
Funcion para calcular los valores y vectores propios
input:
    H: matriz cuadrada hermitica 
output:
    ee: vector con valores propios
    vv: matriz con los vectores propios (para acceder a los vectores usar vv[:,i])
""" 
def Numpyget_eigen(H):
    ee, vv= np.linalg.eigh(H)
    return ee, vv


"""
Funcion para construir el hamiltoniano de la molecula 2-3Ni
input:
    params: lista con los valores de las constantes necesarias para construir el hamiltoniano.
        la lista se compone po J1, J2, J13, h_z, h_x.
output:
    H: matriz del hamiltoniano
""" 
def hamiltoniano(params):
    H = -2*params[0]*Int1 -2*params[1]*Int2 -2*params[2]*Int3 -gyro*nub*params[3]*OZ -gyro*nub*params[4]*OX
    H = np.real(H)
    return H


"""
Matrices de pauli de spin 1
""" 
sI = np.array( [ [1,0,0], [0,1,0], [0,0,1] ] ,dtype='float64')
sX = (1.0/np.sqrt(2))*np.array( [ [0,1,0], [1,0,1], [0,1,0] ], dtype='float64') 
sY = (1.0/np.sqrt(2))*np.array( [ [0,-1j,0], [1j,0,-1j], [0,1j,0] ], dtype='complex64') 
sZ = np.array( [ [1,0,0], [0,0,0], [0,0,-1] ], dtype='float64') 


"""
Estructuras de los acoples
""" 
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

"""
Exchanges moleculares de los dos tipos de moleculas (3D y 1D)
""" 
J3D = [1.49, 1.49, -0.89]
J1D = [-0.08, -0.08, 0.0]
