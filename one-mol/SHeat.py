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
      Phase = []
      for hz in Hz:
            #Construccion del hamiltoniano
            H = hamiltoniano([j1, j2, j3, hz, hx])
            
            #Calculo de valores y vectores propios
            ee, vv= Numpyget_eigen(H)
            
            #Calculo de calor especifico
            valuesbase = [NumpySpecific_heat(ee, t) for t in T]
            Phase.append( valuesbase )

      #Almacenar resultados en un .csv
      Phase = pd.DataFrame( Phase )
      Phase.to_csv("datos/one-mol/sh/"+structure+"hx"+str( hx )+".csv")