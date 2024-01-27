from base import *

"""
Lectura de parametros de la linea de comandos, si es 3D y 1D.
"""
structure = sys.argv[1]
if structure == '3D':
      j1,j2,j3= J3D
else:
      j1,j2,j3= J1D

"""
Variables del sistema, campo magnetico en Z, X y la temperatura
"""
Hz = np.linspace(0.0, 10, 2001)
Hx = np.linspace(0.0, 10, 11)
T = np.linspace(0.01, 10, 1801)

for hx in Hx:
      #Construccion del hamiltoniano
      H = hamiltoniano([j1, j2, j3, 0.0, hx])

      #Calculo de valores y vectores propios
      eeBase, vvBase= Numpyget_eigen(H)

      Phase = []
      for hz in Hz:
            #Construccion del hamiltoniano
            H = hamiltoniano([j1, j2, j3, hz, hx])

            #Calculo de valores y vectores propios
            ee, vv= Numpyget_eigen(H)

            #Calculo de la variacion de entropia (isothermal entropy change) respecto a campo hz 0 
            # y un campo hz variable, el campo hx es fijo durante todo el calculo.
            valuesbase = [NumpyEntropy(eeBase, t) - NumpyEntropy(ee, t) for t in T]
            Phase.append( valuesbase )

      #Almacenar resultados en un .csv
      Phase = pd.DataFrame( Phase )
      Phase.to_csv("datos/one-mol/iso/"+structure+"hx"+str( hx )+".csv")
