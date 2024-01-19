from base import *

structure = sys.argv[1]
conf = sys.argv[2]
if structure == '3D':
      j1,j2,j3= J3D
else:
      j1,j2,j3= J1D
      
Hz = np.linspace(0.0, 10, 1201)
Hx = np.linspace(0.0, 10, 11)
T = np.linspace(1e-2, 10, 1501)[4:]
#exchanges = [-0.5, -0.25, 0.0, 0.25, 0.5]
exchanges = [-0.125, 0.125]

for j in exchanges:
    for hx in Hx:
            H = hamiltonian([j1, j2, j3, j, 0.0, hx, int(conf)])
            eeBase, vvBase= Numpyget_eigen(H)
            Phase = []

            for hz in Hz:
                  H = hamiltonian([j1, j2, j3, j, hz, hx, int(conf)])
                  ee, vv= Numpyget_eigen(H)
                  valuesbase = [NumpyEntropy (eeBase, t) - NumpyEntropy(ee, t) for t in T]
                  Phase.append( valuesbase )

            Phase = pd.DataFrame( Phase )
            Phase.to_csv("datos/two-mol/iso/"+structure+conf+"j"+str(j)+"hx"+str( hx )+".csv")