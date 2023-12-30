from base import *

structure = sys.argv[1]
if structure == '3D':
      j1,j2,j3= J3D
else:
      j1,j2,j3= J1D

Hz = np.linspace(0.0, 15, 2001)
Hx = np.linspace(0.0, 15, 11)
T = np.linspace(1e-3, 15, 1501)

for hx in Hx:
        Phase = []
        for hz in Hz:
                H = hamiltoniano([j1, j2, j3, hz, hx])
                ee, vv= Numpyget_eigen(H)
                valuesbase = np.array( [ NumpySpecific_heat(ee, t) for t in T ] )
                Phase.append( valuesbase )
        Phase = pd.DataFrame( Phase )
        Phase.to_csv("datos/one-mol/sh/"+structure+"hx"+str( hx )+".csv")