from base import *

"""
Lectura de parametros de la linea de comandos, si es 3D y 1D, conf es la variable
para determinar el tipo de acople.
"""
structure = sys.argv[1]
conf = sys.argv[2]
if structure == '3D':
      j1,j2,j3= J3D
else:
      j1,j2,j3= J1D

"""
Variables del sistema, campo magnetico en Z, X, la temperatura y
los valores de exchange entre moleculas
"""
Hz = np.linspace(0.0, 10, 1201)
Hx = np.linspace(0.0, 10, 11)
T = np.linspace(1e-2, 10, 1501)[4:]
exchanges = [-0.125, 0.125]

for j in exchanges:
    for hx in Hx:
            #Construccion del hamiltoniano
            H = hamiltonian([j1, j2, j3, j, 0.0, hx, int(conf)])

            #Calculo de valores y vectores propios
            eeBase, vvBase= Numpyget_eigen(H)
            Phase = []

            for hz in Hz:
                  #Construccion del hamiltoniano
                  H = hamiltonian([j1, j2, j3, j, hz, hx, int(conf)])

                  #Calculo de valores y vectores propios
                  ee, vv= Numpyget_eigen(H)

                  #Calculo de la variacion de entropia (isothermal entropy change) respecto a campo hz 0 
                  # y un campo hz variable, el campo hx es fijo durante todo el calculo.
                  valuesbase = [NumpyEntropy (eeBase, t) - NumpyEntropy(ee, t) for t in T]
                  Phase.append( valuesbase )

            #Almacenar resultados en un .csv
            Phase = pd.DataFrame( Phase )
            Phase.to_csv("datos/two-mol/iso/"+structure+conf+"j"+str(j)+"hx"+str( hx )+".csv")