from base import *

"""
Lectura de parametros de la linea de comandos, conf es la variable
para determinar el tipo de acople.
"""
conf = sys.argv[1]

"""
Variables del sistema, variacion de los exchange intramoleculares 
campo magnetico en Z, X, la temperatura y
los valores de exchange entre moleculas
"""
array1 = np.around(np.linspace(1.49, -0.08, 11),5)
array2 = np.around(np.linspace(-0.89, 0.0, 11),5)
pairs = list(zip(array1, array2))
Hz = np.linspace(0.0, 10, 1501)
Hx = np.linspace(0.0, 10, 11)
T = np.linspace(1e-2, 10, 1201)[4:]
exchanges = [-0.25, 0.25]

for i,(j1,j3) in enumerate(pairs):
      for j in exchanges:
            for hx in Hx:
                  Phase = []
                  for hz in Hz:
                        #Construccion del hamiltoniano
                        H = hamiltonian([j1, j1, j3, j, hz, hx, int(conf)])

                        #Calculo de valores y vectores propios
                        ee, vv= Numpyget_eigen(H)

                        #Calculo de valores esperados del S_z respecto a cada vector propio
                        mag = np.array( [ ((vv[:,k]).T.conj()).dot(OZ).dot(vv[:,k]) for k in range(len(ee))] )

                        #Calculo de magnetizacion
                        valuesbase = [NumpyMagnetization(mag, ee, t) for t in T]
                        Phase.append( valuesbase )

                  #Almacenar resultados en un .csv 
                  Phase = pd.DataFrame( Phase )
                  Phase.to_csv("datos/quenching/mag/step"+str(i)+conf+"j"+str(j)+"hx"+str( hx )+".csv")