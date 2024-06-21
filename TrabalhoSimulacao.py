# Exercício de probabilidade
import scipy.stats as spt
import numpy as np
import matplotlib.pyplot as plt

# 3) Fazendo o gerador de números aleatórios

# fazendo o algoritmo separadamente
"""
p = 0.5

bern_vector = spt.bernoulli.rvs(p, size = 52)
pot_inv_2 = np.array([1 / (2**i) for i in range(1, 53)])
num_aleatorio = np.dot(bern_vector, pot_inv_2)
print(num_aleatorio)
"""
# implementando ele em uma função que gera um vetor de tamanho n
p = 0.5

def vetor_aleatorio(tamanho):
    pot_inv_2 = np.array([1 / (2**i) for i in range(1, 53)])
    vetor_aleatorio = []
    for i in range(1, tamanho+1):
        bern_vector = spt.bernoulli.rvs(p, size = 52)
        num_aleatorio = np.dot(bern_vector, pot_inv_2)
        vetor_aleatorio.append(num_aleatorio)
    vetor_aleatorio = np.array(vetor_aleatorio)
    return vetor_aleatorio

tamanho = 10000
x = vetor_aleatorio(tamanho)
plt.hist(x, bins = 25, density=True, alpha=0.7, color='red', edgecolor='black')
plt.xlabel('Valores')
plt.ylabel('Densidade')
plt.title('Gerando valores inteiramente aleatorios')
plt.show()


# 4) Usando transformações 
# a) transformação quantil de X ~ Expo(1)
# criando um vetor aleatório 

expo = np.log(1 / (1 - x))
plt.hist(expo, bins = 50, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Valores')
plt.ylabel('Densidade')
plt.title('Expo(1) por transformacao quantil')
plt.show()

# b) transformação Box-Muller de X ~ N(0, 1)
# Para fazer Box-Muller precisamos de U ~ Unif(0, 1) e T ~ Expo(1) assim, X = sqrt(2T) * Cos(U)
# criando outro vetor aleatorio

y = np.random.permutation(x)

norm = np.sqrt(-2 * np.log(x)) * np.cos(2 * np.pi * y)
plt.hist(norm, bins = 50, density=True, alpha=0.7, color='darkblue', edgecolor='black')
plt.xlabel('Valores')
plt.ylabel('Densidade')
plt.title('N(0, 1) por transformacao Box-Muller')
plt.show()




    
    