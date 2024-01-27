# Magnetocaloric effect en 3Ni

Repositorio con los codigos para calcular varias cantidades magnetocaloricas para una, dos y un quenching de dos molecules


# Ejecutar codigo
Para ejecutar los codigos, se tiene que estar en el directiorio base del proyecto, para cada tipo se tienen diferentes parametros que se agregan por linea de comandos.

Para una molecula
`
python3 <archivo> <tipo de estructura>
`

Para dos moleculas 
`
python3 <archivo> <tipo de estructura> <configuracion>
`

Para el quenching de dos moleculas
`
python3 <archivo> <configuracion>
`

Los datos calculados se almacenan en una carpeta llamada datos. El tipo de estructura es 3D o 1D, para las configuraciones, puede tomar los valores 1,2 o 3 (lineal, alternada y perpendicular respectivamente).

# Estructuras de los archivos
La parte mas importante de cada carpeta es el base.py en donde se definen todas las funciones para construir y calcular elementos imporantes.

# Editar archivos
La idea de estos archivos es que sirvan como template para estudiar otras estructuras, como es esperable, una de las primeras cosas que se tienen que cambiar, son las funciones y elementos usandos para la construccion del hamiltoniano, lo siguiente es editar el flujo para que reciba los parametros de la nueva estructura y itere sobre los nuevos rangos y cantidades.