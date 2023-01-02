import streamlit as st

st.markdown('''
### Outliers
Il y a plusieurs façons de trouver les valeurs aberrantes (aussi appelées "outliers") dans une variable quantitative. Voici quelques approches courantes :

Utiliser un diagramme en boîte et des moustaches (aussi appelé "boîte à moustaches") : ce type de diagramme permet de visualiser la distribution des données et de repérer les valeurs aberrantes qui sont situées en dehors des moustaches.

Utiliser la méthode du quantile : pour chaque quantile (par exemple, le premier quartile, le deuxième quartile, etc.), on peut déterminer un seuil au-delà duquel les valeurs seront considérées comme aberrantes. Par exemple, si le premier quartile est Q1 et le troisième quartile est Q3, alors les valeurs situées en dehors de l'intervalle [Q1 - 1,5 * (Q3 - Q1), Q3 + 1,5 * (Q3 - Q1)] peuvent être considérées comme des outliers. Cette approche est souvent appelée "méthode de Tukey".

Utiliser des tests statistiques : il existe plusieurs tests statistiques qui permettent de détecter les valeurs aberrantes dans une distribution, par exemple le test de Grubbs ou le test de D'Agostino. Ces tests sont basés sur l'hypothèse que les données suivent une certaine distribution (par exemple, une distribution normale), et qu'il est possible de déterminer un seuil au-delà duquel les valeurs sont considérées comme aberrantes.

Il est important de noter que la définition même de ce qu'est un outlier peut varier selon le contexte et les données en question. Par conséquent, il est recommandé de choisir l'approche la plus adaptée à votre cas d'utilisation et de vérifier si les valeurs détectées comme aberrantes sont effectivement pertinentes dans le contexte de votre étude.



Voici un exemple de la deuxième méthode que j'ai mentionnée, c'est-à-dire la méthode du quantile, en utilisant Python :

''')

code = r"""
import numpy as np

# On suppose que les données sont stockées dans un tableau numpy nommé "data"

# Calcul des quantiles
q1, q3 = np.percentile(data, [25, 75])

# Calcul du seuil à partir des quantiles
threshold = 1.5 * (q3 - q1)

# Détection des outliers
outliers = data[(data < q1 - threshold) | (data > q3 + threshold)]

# Affichage des outliers
print(outliers)

"""

st.code(code, language='python')
st.markdown("""Ce code calcule le premier et le troisième quantile des données, puis utilise ces valeurs pour déterminer le seuil à partir duquel les valeurs seront considérées comme aberrantes (selon la méthode de Tukey). Ensuite, il sélectionne les valeurs du tableau data qui sont situées en dehors de cet intervalle et les affiche à l'écran.

Il est important de noter que cette approche ne convient pas à tous les cas d'utilisation et qu'il peut être nécessaire de faire des ajustements en fonction de la distribution des données et de l'objectif de l'analyse.""")