import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.markdown(r"""
# Mesures de concentration 
## sont le plus souvent utilisées pour des sommes d'argent ! Étudier la concentration d'argent, c'est regarder si l'argent est réparti de manière égalitaire ou pas.

Ce que l'on va regarder, c'est si tout l'argent que vous dépensez se concentre en quelques opérations bancaires, ou si au contraire, il est bien réparti parmi les opérations. Dire que votre argent se concentre sur quelques opérations signifie que généralement, vous faites de très nombreuses petites dépenses, et que parfois, il vous arrive de faire quelques dépenses énormes.

Au contraire, l'argent que vous dépensez est bien réparti si toutes vos opérations bancaires (sortantes) ont à peu près le même montant.

Pour visualiser cela, nous utilisons la `courbe de Lorenz`
#### l'indice de Gini
La courbe de Lorenz n'est pas une statistique, c'est une courbe ! Du coup, on a créé l'indice de Gini, qui résume la courbe de Lorenz.

Il mesure l'aire présente entre la première bissectrice et la courbe de Lorenz. Plus précisément, si on note S cette aire, alors :

gini=2 × S


L'indice de Gini est un indicateur statistique utilisé pour mesurer l'égalité dans une société ou une distribution. Plus précisément, l'indice de Gini mesure la dispersion des valeurs autour de la médiane d'une distribution, en utilisant une échelle de 0 à 1, où 0 représente une distribution parfaite de l'égalité (c'est-à-dire une situation où chaque individu dans la distribution a la même valeur) et 1 représente une distribution parfaite de l'inégalité (c'est-à-dire une situation où un seul individu dans la distribution a toutes les valeurs et tous les autres n'ont aucune valeur).

L'indice de Gini est souvent utilisé pour mesurer l'inégalité des revenus dans une économie, mais il peut également être utilisé pour mesurer l'inégalité dans d'autres domaines, tels que la distribution de la richesse, la distribution de la santé ou la distribution de l'accès à l'éducation.


On va utiliser l'indice de Gini pour mesurer l'inégalité dans la distribution des assurances. Par exemple, si vous souhaitez savoir si les personnes d'une certaine région ont un accès égal aux assurances de qualité et à des tarifs abordables, vous pouvez utiliser l'indice de Gini pour mesurer l'inégalité dans la distribution de l'accès aux assurances dans cette région. Si l'indice de Gini est élevé, cela indique qu'il y a une forte inégalité dans l'accès aux assurances dans cette région, ce qui pourrait être dû à des facteurs tels que la santé, le revenu, l'emplacement géographique ou d'autres facteurs.

Il est important de noter que l'indice de Gini ne mesure que la dispersion des valeurs autour de la médiane, il ne tient pas compte de la façon dont la distribution est répartie au-dessus ou en dessous de la médiane. Par conséquent, il peut être utile de combiner l'indice de Gini avec d'autres indicateurs pour avoir une image complète de l'inégalité dans la distribution des assurances.

""")

code = """
data = pd.read_csv('data.csv')
dep = data['charges']
n = len(dep)
lorenz = np.cumsum(np.sort(dep)) / dep.sum()
lorenz = np.append([0],lorenz) # La courbe de Lorenz commence à 0

xaxis = np.linspace(0-1/n,1+1/n,n+1) #Il y a un segment de taille n pour chaque individu, plus 1 segment supplémentaire d'ordonnée 0. Le premier segment commence à 0-1/n, et le dernier termine à 1+1/n.
plt.plot(xaxis,lorenz,drawstyle='steps-post')

xaxis = np.linspace(0-1/n,1+1/n,len(lorenz)) #Il y a un segment de taille n pour chaque individu, plus 1 segment supplémentaire d'ordonnée 0. Le premier segment commence à 0-1/n, et le dernier termine à 1+1/n.
plt.plot(xaxis,lorenz,drawstyle='steps-post')
plt.plot([0,1], [0,1]) #tracer la bisséctrice

# plt.show()
AUC = (lorenz.sum() -lorenz[-1]/2 -lorenz[0]/2)/n # Surface sous la courbe de Lorenz. Le premier segment (lorenz[0]) est à moitié en dessous de 0, on le coupe donc en 2, on fait de même pour le dernier segment lorenz[-1] qui est à moitié au dessus de 1.
S = 0.5 - AUC # surface entre la première bissectrice et le courbe de Lorenz
gini = 2*S
# another methode : 
def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))
print(f"gini coefiicient : {gini(dep)}")
"""


st.code(code, language="python")

data = pd.read_csv('data.csv')
dep = data['charges']
n = len(dep)
lorenz = np.cumsum(np.sort(dep)) / dep.sum()
lorenz = np.append([0],lorenz) # La courbe de Lorenz commence à 0

xaxis = np.linspace(0-1/n,1+1/n,n+1) #Il y a un segment de taille n pour chaque individu, plus 1 segment supplémentaire d'ordonnée 0. Le premier segment commence à 0-1/n, et le dernier termine à 1+1/n.
fig, ax = plt.plot(xaxis,lorenz,drawstyle='steps-post')

xaxis = np.linspace(0-1/n,1+1/n,len(lorenz)) #Il y a un segment de taille n pour chaque individu, plus 1 segment supplémentaire d'ordonnée 0. Le premier segment commence à 0-1/n, et le dernier termine à 1+1/n.
plt.plot(xaxis,lorenz,drawstyle='steps-post')
plt.plot([0,1], [0,1]) #tracer la bisséctrice
st.pyplot(fig)

# plt.show()
AUC = (lorenz.sum() -lorenz[-1]/2 -lorenz[0]/2)/n # Surface sous la courbe de Lorenz. Le premier segment (lorenz[0]) est à moitié en dessous de 0, on le coupe donc en 2, on fait de même pour le dernier segment lorenz[-1] qui est à moitié au dessus de 1.
S = 0.5 - AUC # surface entre la première bissectrice et le courbe de Lorenz
gini = 2*S
# another methode : 
def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))
