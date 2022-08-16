# Run SQL queries in EXCEL

# Contents of ~/my_app/streamlit_app.py
from cProfile import label
import streamlit as st

st.set_page_config(
    page_icon=":blue_heart:",
)


st.markdown("<h1 style='text-align: center; color: black;'>Run SQL queries in EXCEL</h1>", unsafe_allow_html=True)
st.markdown(r"""
you'll learn how to add the SQL formula to your Excel application. In particular, we will use the Python package `xlwings` and the xlwings Excel add-in. 

### Refrences :
* [Youtube](https://www.youtube.com/watch?v=pXbvHdWRlJM&t=169s)
* [github](https://github.com/Sven-Bo/run-sql-queries-in-excel)

### Requirements
Install the dependencies with pip
""")

code = """
pip install xlwings
"""
st.code(code, language='python')


st.markdown(r"""
Install the xlwings Excel add-in
""")
code = """
xlwings addin install
"""
st.code(code, language='python')


st.markdown(r"""
### Excel document
Download the following [excel file](https://github.com/Sven-Bo/run-sql-queries-in-excel/blob/master/SQL_In_Excel.xlsx?raw=true) to start practicing 

In this excel document you have three tabels :
* `books` which will be considered  later as tabel `a`
* `authors` which will be considered  later as tabel `b`
* `editors` which will be considered  later as tabel `c`

Now start using the sql queries you can find from line 29. 
Then use the fonction `=sql([query]; [table a]; [tabel b]; [tabel c]; ...)`

### Examples :
##### Example 1
""")
code = """
=sql(SELECT id, title FROM a WHERE language = 'german' ; A2:E10)
"""

st.code(code, language='excel')


from PIL import Image
import os
os.chdir(r"C:\Users\emada\PycharmProjects\Mylearning\Streamlit\pages\streamlitML\pages")
image = Image.open('Excel.jpg')

st.image(image, caption='')


st.markdown(r"""
where `A2:E10` is table `a`

##### Example 2
""")
code = """
=sql(SELECT 
        a.id, a.title, b.last_name AS author_name , c.last_name AS editor_name 
    FROM 
        a INNER JOIN b 
    ON 
        a.author_id = b.id 
    INNER JOIN c 
        ON a.editor_id = c.id
    ; A2:E10; G2:I7; K2:M9)
"""

st.code(code, language='excel')


st.markdown(r"""
where :
- `A2:E10` is table `a`
- `G2:I7` is table `b`
- `K2:M9` is table `c`
""")


st.markdown(r"""
### Note

You should write the query in different cell then call it in `=sql()` fonction
""")