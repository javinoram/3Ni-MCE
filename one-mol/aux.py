from base import *

structure = sys.argv[1]
if structure == '3D':
      j1,j2,j3= J3D
else:
      j1,j2,j3= J1D

Hz = np.linspace(0.0, 1, 101)
T = [5.0]

Phase = []
for hz in Hz:
    H = hamiltoniano([j1, j2, j3, hz, 0.0])
    ee, vv= Numpyget_eigen(H)
    mag = np.array( [ ((vv[:,k]).T.conj()).dot(OZ).dot(vv[:,k]) for k in range(len(ee))] )
    valuesbase = [NumpyMagnetization(mag, ee, t) for t in T]
    Phase.append( valuesbase )
Phase = pd.DataFrame( Phase )
Phase.to_csv("special.csv")