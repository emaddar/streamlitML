import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

st.set_page_config(
    page_icon=":blue_heart:"
)




st.markdown("<h1 style='text-align: center; color: black;'>seaborn</h1>", unsafe_allow_html=True)


st.markdown(r"""
[Seaborn](https://seaborn.pydata.org/) is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

##### Installing from PyPI
In terminal :
""")
st.code("pip install seaborn", language='python')

st.markdown(r"""
##### Import mathplotlib
""")
st.code("import seaborn as sns", language='python')


