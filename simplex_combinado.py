import numpy as np

# Funciones auxiliares, solo para imprimir de forma clara (creadas con IA, debido a que se trata de 
# un aspecto ajeno al método simplex, y no es el objetivo de la práctica)
def _titulo(texto, caracter="="):
    print("\n" + caracter * 72)
    print(f" {texto}")
    print(caracter * 72)
 
 
def _mostrar_tableau(matriz, basicas):
    """Imprime el tableau con cabecera de columnas y etiqueta de cada fila."""
    num_variables = matriz.shape[1] - 2
    cabecera = ["z"] + [f"x{i}" for i in range(1, num_variables + 1)] + ["b"]
    filas = ["z"] + [f"x{v}" for v in basicas]
 
    print("        " + "".join(f"{c:>10}" for c in cabecera))
    print("        " + "-" * (10 * len(cabecera)))
    for etiqueta, fila in zip(filas, matriz):
        celdas = "".join(f"{round(float(v), 4) + 0.0:>10.3f}" for v in fila)
        print(f"  {etiqueta:<4} |{celdas}")


def ejecutar_simplex_primal(matriz_simplex: np.array, basicas=None, no_basicas=None, matriz_simplex_original=None):

    """
    Ejecuta el método simplex primal para resolver un problema de programación lineal.
    
    Parámetros:
    matriz_simplex (np.array): La matriz del tableau simplex.
    basicas, no_basicas (list): Base actual (opcional, se modifican in situ).
    matriz_simplex_original (np.array): Matriz símplex de referencia (opcional).
    
    Retorna:
    np.array: La solución óptima del problema.
    """

    
    # 1) Inicialización de variables 
    matriz_simplex = matriz_simplex.astype(float)  # Aseguramos que la matriz sea de tipo float 
    if matriz_simplex_original is None:
        matriz_simplex_original = matriz_simplex.copy()  # Guardamos la matriz original para referencia
    num_variables = matriz_simplex.shape[1] - 2  # Número de variables (excluyendo la columna de soluciones)
    num_restricciones = matriz_simplex[1:].shape[0]  # Número de restricciones (excluyendo la fila de la función objetivo)

    # 2) Inicialización de las variables básicas y no básicas
    if basicas is None or no_basicas is None:
        # Obtenemos las variables básicas como las últimas variables en la matriz simplex según el número de restricciones
        basicas = list(range(num_variables - num_restricciones + 1, num_variables + 1))
        # Obtenemos las variables no básicas de la misma manera, pero excluyendo las variables básicas
        no_basicas = list(range(1, num_variables - num_restricciones + 1))

    _titulo("FASE PRIMAL (se busca optimalidad)")
    print("Variables básicas iniciales:", basicas)
    print("Variables no básicas iniciales:", no_basicas)


    # 3) Bucle de iteraciones
    num_iter = 0
    while (matriz_simplex[0, 1:-1] < 0).any():

        # 3.1) Mostrar información de la iteración actual
        num_iter += 1
        _titulo(f"PRIMAL · Iteración {num_iter}", "-")
        _mostrar_tableau(matriz_simplex, basicas)

        # 3.2) Establecemos variable de entrada
        arg_entrada = matriz_simplex[0, no_basicas].argmin()
        var_entrada = no_basicas[arg_entrada]

        # 3.3) Establecemos variable de salida
        # Importante: Solo cuentan los a_je > 0 (min{ b_j / a_je | a_je > 0 }),
        # tal como dice el algoritmo. Antes, con a_je < 0 y b_j = 0 salía -0.0,
        # que no es < 0 y se colaba como "mínimo".
        columna = matriz_simplex[1:, var_entrada]
        with np.errstate(divide="ignore", invalid="ignore"): # Se silencian las advertencias de división por cero y valores inválidos
            cocientes = matriz_simplex[1:, -1] / columna
        cocientes[columna < 0] = np.inf  # Ignoramos los cocientes negativos
        arg_salida = cocientes.argmin()
        var_salida = basicas[arg_salida]

        print(f"\n  >> Entra x{var_entrada}   |   Sale x{var_salida}")

        # 3.4) Actualizamos las listas de variables básicas y no básicas
        basicas[arg_salida] = var_entrada
        no_basicas[arg_entrada] = var_salida

        # 3.5) Realizamos las operaciones pertinentes de pivoteo

        # 3.5.1) Calculamos los coeficientes de las variables básicas y no básicas
        coeficientes_basicas = - matriz_simplex_original[0, basicas]
        coeficientes = - matriz_simplex_original[0, 1:num_variables+1]

        matriz_basicas = matriz_simplex_original[1:, basicas]
        matriz_basicas_inv = np.linalg.inv(matriz_basicas)

        matriz_restricciones = matriz_simplex_original[1:, 1:num_variables+1]
        b = matriz_simplex_original[1:, -1]

        # 3.5.2) Realizamos los cálculos necesarios
        b_inv_a = matriz_basicas_inv @ matriz_restricciones
        b_inv_b = matriz_basicas_inv @ b

        nuevos_coeficientes = coeficientes_basicas @ b_inv_a - coeficientes
        nuevo_total = coeficientes_basicas @ b_inv_b

        # 3.5.3) Creamos la nueva matriz simplex con los valores actualizados
        nueva_matriz_simplex = np.zeros_like(matriz_simplex_original)
        nueva_matriz_simplex[0, 0] = 1
        nueva_matriz_simplex[0, 1:num_variables+1] = nuevos_coeficientes
        nueva_matriz_simplex[0, -1] = nuevo_total
        nueva_matriz_simplex[1:, 1:num_variables+1] = b_inv_a
        nueva_matriz_simplex[1:, -1] = b_inv_b

        matriz_simplex = nueva_matriz_simplex

    _titulo("PRIMAL · Tableau final", "-")
    _mostrar_tableau(matriz_simplex, basicas)

    return matriz_simplex



def ejecutar_simplex_dual(matriz_simplex: np.array, basicas=None, no_basicas=None, matriz_simplex_original=None):

    """
    Ejecuta el método simplex dual para resolver un problema de programación lineal.
    
    Parámetros:
    matriz_simplex (np.array): La matriz del tableau simplex.
    basicas, no_basicas (list): Base actual (opcional, se modifican in situ).
    matriz_simplex_original (np.array): Matriz símplex de referencia (opcional).
    
    Retorna:
    np.array: La solución óptima del problema.
    """

    # 1) Inicialización de variables 
    matriz_simplex = matriz_simplex.astype(float) # Aseguramos que la matriz sea de tipo float
    if matriz_simplex_original is None:
        matriz_simplex_original = matriz_simplex.copy()  # Guardamos la matriz original para referencia
    num_variables = matriz_simplex.shape[1] - 2  # Número de variables (excluyendo la columna de soluciones)
    num_restricciones = matriz_simplex[1:].shape[0]  # Número de restricciones (excluyendo la fila de la función objetivo)

    # 2) Inicialización de las variables básicas y no básicas
    if basicas is None or no_basicas is None:
        # Obtenemos las variables básicas como las últimas variables en la matriz simplex según el número de restricciones
        basicas = list(range(num_variables - num_restricciones + 1, num_variables + 1))
        # Obtenemos las variables no básicas de la misma manera, pero excluyendo las variables básicas
        no_basicas = list(range(1, num_variables - num_restricciones + 1))

    _titulo("FASE DUAL (se busca factibilidad: todos los b_j >= 0)")
    print(f"  Variables básicas iniciales:    {basicas}")
    print(f"  Variables no básicas iniciales: {no_basicas}")


    # 3) Bucle de iteraciones
    num_iter = 0
    while (matriz_simplex[1:, -1] < 0).any():

        # 3.1) Mostrar información de la iteración actual
        num_iter += 1
        _titulo(f"DUAL · Iteración {num_iter}", "-")
        _mostrar_tableau(matriz_simplex, basicas)

        # 3.2) Establecemos variable de salida
        arg_salida = matriz_simplex[1:, -1].argmin()
        var_salida = basicas[arg_salida]

        # 3.3) Establecemos variable de entrada
        coeficientes_fila = matriz_simplex[arg_salida + 1, no_basicas]
        candidatas = coeficientes_fila < 0

        # Escogemos los coeficientes de la fila de la variable de salida que son negativos
        indices_candidatos = np.flatnonzero(candidatas)
        costes_reducidos = matriz_simplex[0, no_basicas]

        cocientes = np.abs(
            costes_reducidos[indices_candidatos]
            / coeficientes_fila[indices_candidatos]
        )

        arg_entrada = indices_candidatos[cocientes.argmin()]
        var_entrada = no_basicas[arg_entrada]

        print(f"\n  >> Sale x{var_salida}   |   Entra x{var_entrada}")

        # 3.4) Actualizamos las listas de variables básicas y no básicas
        basicas[arg_salida] = var_entrada
        no_basicas[arg_entrada] = var_salida

        # 3.5) Realizamos las operaciones pertinentes de pivoteo

        # 3.5.1) Calculamos los coeficientes de las variables básicas y no básicas
        coeficientes_basicas = - matriz_simplex_original[0, basicas]
        coeficientes = - matriz_simplex_original[0, 1:num_variables+1]

        matriz_basicas = matriz_simplex_original[1:, basicas]
        matriz_basicas_inv = np.linalg.inv(matriz_basicas)

        matriz_restricciones = matriz_simplex_original[1:, 1:num_variables+1]
        b = matriz_simplex_original[1:, -1]

        # 3.5.2) Realizamos los cálculos necesarios
        b_inv_a = matriz_basicas_inv @ matriz_restricciones
        b_inv_b = matriz_basicas_inv @ b

        nuevos_coeficientes = coeficientes_basicas @ b_inv_a - coeficientes
        nuevo_total = coeficientes_basicas @ b_inv_b

        # 3.5.3) Creamos la nueva matriz simplex con los valores actualizados
        nueva_matriz_simplex = np.zeros_like(matriz_simplex_original)
        nueva_matriz_simplex[0, 0] = 1
        nueva_matriz_simplex[0, 1:num_variables+1] = nuevos_coeficientes
        nueva_matriz_simplex[0, -1] = nuevo_total
        nueva_matriz_simplex[1:, 1:num_variables+1] = b_inv_a
        nueva_matriz_simplex[1:, -1] = b_inv_b

        matriz_simplex = nueva_matriz_simplex

    _titulo("DUAL · Tableau final", "-")
    _mostrar_tableau(matriz_simplex, basicas)

    return matriz_simplex



def ejecutar_simplex_combinado(matriz_simplex: np.array, objetivo="max"): 

    """
    Ejecuta el método simplex combinado para resolver un problema de programación lineal.
    
    Parámetros:
    matriz_simplex (np.array): La matriz del tableau simplex.
    objetivo (str): "max" para maximización, "min" para minimización.
    
    Retorna:
    np.array: La solución óptima del problema.
    """
    # 1) Inicialización de variables
    matriz_simplex = matriz_simplex.astype(float)
    num_variables = matriz_simplex.shape[1] - 2
    num_restricciones = matriz_simplex[1:].shape[0]

    # 2) Se crea la base y se comparte entre dual y primal (las listas se modifican 
    # in situ dentro de cada función)
    basicas = list(range(num_variables - num_restricciones + 1, num_variables + 1))
    no_basicas = list(range(1, num_variables - num_restricciones + 1))

    # 3) Se analiza si el origen (0, 0) se encuentra en la región factible, 
    # para lo cual se verifica si todas las variables básicas tienen valores no negativos
    if (matriz_simplex[1:, -1] < 0).any():
        _titulo(f"SIMPLEX COMBINADO ({objetivo}) · el origen NO es factible: dual -> primal", "#")

        # Se pasa a un problema de minimización
        if objetivo == "max":
            matriz_simplex[0, 1:] *= -1  
        matriz_original = matriz_simplex.copy()

        # Iteraciones del dual
        matriz_simplex = ejecutar_simplex_dual(matriz_simplex, basicas, no_basicas, matriz_original)

        # En este punto, todos los b_j >= 0, así que se multiplica
        # el renglón z por -1 (tanto en la matriz actual como en la de
        # referencia, porque el pivoteo recalcula desde esta última) y se
        # continúa con el primal.
        matriz_simplex[0, 1:] *= -1
        matriz_original[0, 1:] *= -1

    # 4) Si todas las variables básicas tienen valores no negativos, usamos el método primal
    else:
        _titulo(f"SIMPLEX COMBINADO ({objetivo}) · el origen es factible: primal", "#")

        # Se pasa a un problema de maximización
        if objetivo == "min":
            matriz_simplex[0, 1:] *= -1  
        matriz_original = matriz_simplex.copy()
 
    # Iteraciones del primal (si ya es óptimo, no itera)
    matriz_simplex = ejecutar_simplex_primal(matriz_simplex, basicas, no_basicas, matriz_original)

    # 5) Devolver el problema a su versión original.
    # Si era de minimización, se deshace el cambio de signo de la fila z,
    # para que el valor de z que se lee sea el del problema original.
    if objetivo == "min":
        matriz_simplex[0, 1:] *= -1

    # 6) Resumen de la solución (hecho con IA)
    _titulo("SOLUCIÓN ÓPTIMA (problema original)", "#")
    solucion = np.zeros(num_variables)
    for fila, var in enumerate(basicas):
        solucion[var - 1] = matriz_simplex[fila + 1, -1]
    for i, valor in enumerate(solucion, start=1):
        print(f"  x{i} = {round(float(valor), 6) + 0.0:.4f}")
    print(f"\n  z ({objetivo}) = {round(float(matriz_simplex[0, -1]), 6) + 0.0:.4f}")