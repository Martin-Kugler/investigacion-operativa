import numpy as np


def ejecutar_simplex_primal(matriz_simplex: np.array, objetivo="max"):

    """
    Ejecuta el método simplex primal para resolver un problema de programación lineal.
    
    Parámetros:
    matriz_simplex (np.array): La matriz del tableau simplex.
    objetivo (str): "max" para maximización, "min" para minimización.
    
    Retorna:
    np.array: La solución óptima del problema.
    """
    
    # Implementación del método simplex primal
    # Aquí iría la lógica para realizar las iteraciones del método simplex
    # y encontrar la solución óptima.
    
    # Por ahora, devolvemos la matriz original como un placeholder
    return matriz_simplex

def ejecutar_simplex_dual(matriz_simplex: np.array):

    """
    Ejecuta el método simplex dual para resolver un problema de programación lineal.
    
    Parámetros:
    matriz_simplex (np.array): La matriz del tableau simplex.
    
    Retorna:
    np.array: La solución óptima del problema.
    """

    # 1) Inicialización de variables 
    matriz_simplex_original = matriz_simplex.copy()  # Guardamos la matriz original para referencia
    num_variables = matriz_simplex.shape[1] - 2  # Número de variables (excluyendo la columna de soluciones)
    num_restricciones = matriz_simplex[1:].shape[0]  # Número de restricciones (excluyendo la fila de la función objetivo)

    # 2) Inicialización de las variables básicas y no básicas
    # Obtenemos las variables básicas como las últimas variables en la matriz simplex según el número de restricciones
    basicas = list(range(num_variables - num_restricciones + 1, num_variables + 1))
    # Obtenemos las variables no básicas de la misma manera, pero excluyendo las variables básicas
    no_basicas = list(range(1, num_variables - num_restricciones + 1))

    print("Variables básicas iniciales:", basicas)
    print("Variables no básicas iniciales:", no_basicas)


    # 3) Bucle de iteraciones
    num_iter = 0
    while (matriz_simplex[1:, -1] < 0).any():

        # 3.1) Mostrar información de la iteración actual
        num_iter += 1
        print(f"\nIteración {num_iter}:")

        print("\n\nMatriz simplex actual:")
        print(matriz_simplex)

        print("\nVariables básicas:", basicas)
        print("\nVariables no básicas:", no_basicas)

        # 3.2) Establecemos variable de salida
        arg_salida = matriz_simplex[1:, -1].argmin()
        var_salida = basicas[arg_salida]

        # 3.3)Establecemos variable de entrada
        arg_entrada = np.abs(matriz_simplex[0, no_basicas] / matriz_simplex[arg_salida+1, no_basicas]).argmin()
        var_entrada = no_basicas[arg_entrada]

        print(f"\n\nVariable de salida: {var_salida}")
        print(f"\nVariable de entrada: {var_entrada}")

        # 3.4) Actualizamos las listas de variables básicas y no básicas
        basicas[arg_salida] = var_entrada
        basicas.sort()

        no_basicas[arg_entrada] = var_salida
        no_basicas.sort()

        print("\n\nVariables básicas actualizadas:", basicas)
        print("\nVariables no básicas actualizadas:", no_basicas)

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

        print("\n\nNueva matriz simplex:")
        print(matriz_simplex)

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

    # Se analiza si el origen (0, 0) se encuentra en la región factible, 
    # para lo cual se verifica si todas las variables básicas tienen valores no negativos
    if (matriz_simplex[1:, -1] < 0).any():
        if objetivo == "max":
            matriz_simplex[0] = matriz_simplex[0] * -1  
            return ejecutar_simplex_dual(matriz_simplex)
        else:
            return ejecutar_simplex_dual(matriz_simplex)

    # Si todas las variables básicas tienen valores no negativos, usamos el método primal
    else: 
        if objetivo == "min":
            matriz_simplex[0] = matriz_simplex[0] * -1  
            return ejecutar_simplex_primal(matriz_simplex, objetivo="max")
        else: 
            return ejecutar_simplex_primal(matriz_simplex, objetivo)