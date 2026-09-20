"""
Script auxiliar que define las clases y etiquetas utilizadas en el proyecto. 
Sus variables se utilizan en varios scripts a lo largo de la ejecución
"""

# Nombres de las clases
CLASSES = ["Wake", "Sleep"]  
# Número de clases 
N_CLASSES = len(CLASSES) 
# Etiquetas
LABELS = list(range(N_CLASSES)) 
# Diccionario que relaciona etiqueta con nombre de la clase
CLASS_NAMES = {i: name for i, name in enumerate(CLASSES)} 
