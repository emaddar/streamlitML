import streamlit as st
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt

# Générer deux séries de données aléatoires
np.random.seed(123)
x = np.random.normal(size=100)
y = np.random.normal(size=100)

# Calculer les corrélations de Pearson, Spearman et Kendall
r_pearson, p_pearson = pearsonr(x, y)
r_spearman, p_spearman = spearmanr(x, y)
tau, p_kendall = kendalltau(x, y)

# Tracer un diagramme de dispersion et ajouter la régression linéaire OLS
plt.scatter(x, y)
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color="r")

# Afficher le graphique
st.pyplot()

# Afficher les résultats des tests de corrélation
st.write(f"Corrélation de Pearson entre x et y : {r_pearson:.2f} (p = {p_pearson:.3f})")
st.write(f"Corrélation de Spearman entre x et y : {r_spearman:.2f} (p = {p_spearman:.3f})")
st.write(f"Corrélation de Kendall entre x et y : {tau:.2f} (p = {p_kendall:.3f})")

# Interpréter les résultats des tests de corrélation
st.write("Interprétation :")
if p_pearson < 0.05:
    st.write("Il y a une relation significativement positive entre x et y selon le test de Pearson.")
else:
    st.write("Il n'y a pas de relation significativement positive entre x et y selon le test de Pearson.")

if p_spearman < 0.05:
    st.write("Il y a une relation significativement positive entre x et y selon le test de Spearman.")
else:
    st.write("Il n'y a pas de relation significativement positive entre x et y selon le test de Spearman.")

if p_kendall < 0.05:
    st.write("Il y a une relation significativement positive entre x et y selon le test de Kendall.")
else:
    st.write("Il n'y a pas de relation significativement positive entre x et y selon le test de kendall")



st.markdown("""# with statmodels package :""")

import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Générer deux séries de données aléatoires
np.random.seed(123)
x = np.random.normal(size=100)
y = np.random.normal(size=100)

# Créer un DataFrame avec les données x et y
df = pd.DataFrame({"x": x, "y": y})

# Effectuer une régression linéaire OLS avec x comme variable explicative et y comme variable à expliquer
model = smf.ols("y ~ x", data=df)
result = model.fit()

# Afficher le résultat sous forme de tableau
st.write(result.summary())