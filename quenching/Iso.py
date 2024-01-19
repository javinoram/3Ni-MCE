from base import *

conf = sys.argv[1]
array1 = np.around(np.linspace(1.49, -0.08, 11),5)
array2 = np.around(np.linspace(-0.89, 0.0, 11),5)
pairs = list(zip(array1, array2))


Hz = np.linspace(0.0, 10, 1201)
Hx = np.linspace(0.0, 10, 11)
T = np.linspace(1e-2, 10, 1501)[4:]
exchanges = [-0.25, 0.25]

for i,(j1,j3) in enumerate(pairs):
      for j in exchanges:
            for hx in Hx:
                  print(i, j1, j3, j, hx)
                  H = hamiltonian([j1, j1, j3, j, 0.0, hx, int(conf)])
                  eeBase, vvBase= Numpyget_eigen(H)
                  Phase = []

                  for hz in Hz:
                        H = hamiltonian([j1, j1, j3, j, hz, hx, int(conf)])
                        ee, vv= Numpyget_eigen(H)
                        valuesbase = [NumpyEntropy (eeBase, t) - NumpyEntropy(ee, t) for t in T]
                        Phase.append( valuesbase )

                  Phase = pd.DataFrame( Phase )
                  Phase.to_csv("datos/quenching/iso/step"+str(i)+conf+"j"+str(j)+"hx"+str( hx )+".csv")