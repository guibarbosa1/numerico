import numpy as np
import matplotlib.pyplot as plt

def ajuste_minimos_quadrados(x, y, grau):
    # Construir a matriz de Vandermonde
    X = np.vander(x, grau + 1, increasing=True)
    
    # Calcular os coeficientes pelo método dos mínimos quadrados
    coeficientes, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    
    return coeficientes

def plotar_ajuste(x, y, coeficientes):
    # Gerar pontos para o polinômio ajustado
    xajuste = np.linspace(min(x), max(x), 100)
    yajuste = np.polyval(coeficientes, xajuste)
    
    # Plotar os dados originais e o ajuste
    plt.scatter(x, y, label='Dados Originais')
    plt.plot(xajuste, yajuste, label='Ajuste Polinomial', color='red')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

# Dados de exemplo
x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
y = np.array([-0.609, -0.317, -0.148, 0.235, -0.447, -0.903, -0.602, -0.881, 0.701, -0.99, -0.091])

# Grau do polinômio
grau_polinomio = 9

# Ajuste pelo método dos mínimos quadrados
coeficientes_ajuste = ajuste_minimos_quadrados(x, y, grau_polinomio)

# Imprimir os coeficientes
print(f'Coeficientes do ajuste: {coeficientes_ajuste}')

# Plotar o ajuste
plotar_ajuste(x, y, coeficientes_ajuste)

