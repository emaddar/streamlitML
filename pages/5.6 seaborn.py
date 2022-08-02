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





st.markdown(r"""
##### Load Dataset
""")
code = '''
df_tips = sns.load_dataset('tips')  # This dataset is embedded in seaborn library
print(df_tips.shape)  # (244, 7)
'''
st.code(code, language='python')

st.write('[informations](https://www.rdocumentation.org/packages/reshape2/versions/1.4.2/topics/tips) about `tips` dataset.')
st.write("Find more data for seaborn [here](https://github.com/mwaskom/seaborn-data).")

import matplotlib.pyplot as plt
import seaborn as sns

df_tips = sns.load_dataset('tips')  # This dataset is embedded in seaborn library
print(df_tips.shape)  # (244, 7)

st.code("print(df_tips.head())")
st.dataframe(df_tips.head())