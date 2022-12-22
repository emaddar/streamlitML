import streamlit as st

st.set_page_config(
    page_icon=":blue_heart:",
)


#python -m pip install sidetable
import pandas as pd 


code = r"""
df = pd.DataFrame({'A': ['a', 'b', 'a', 'b'],
'B': [1, 2, 3, 4],
'C': ['x', 'y', 'z', 'w']})
df.stb.freq(['A'])
#Or

df.stb.freq(['A','B'])
"""

st.code(code, language='Python')