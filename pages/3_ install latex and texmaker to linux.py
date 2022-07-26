# Run SQL queries in EXCEL

# Contents of ~/my_app/streamlit_app.py
from cProfile import label
import streamlit as st

st.set_page_config(
    page_icon=":blue_heart:",
)

st.image("data:image/png;base64,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")
st.markdown('To install `LaTex` in linux (**This may take alot of time**):' )

code = "sudo apt install texlive-full"
st.code(code)





st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRw_Cg2JYBDbAUe7fsUpkqoru_kIzERqd7fXIBqVuGHRaHzq_KL7bB9ZhxDtp05geQJYpY&usqp=CAU")
st.markdown('To install `TexMaker` in linux ' )

code = "sudo apt install texmaker"
st.code(code)

st.markdown("""
For more, watch [this youtube](https://www.youtube.com/watch?v=ioy7e7fxjHo)
""")