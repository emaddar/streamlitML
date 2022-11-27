# Run SQL queries in EXCEL

# Contents of ~/my_app/streamlit_app.py
from cProfile import label
import streamlit as st

st.set_page_config(
    page_icon=":blue_heart:",
)

st.image("https://res.cloudinary.com/dyd911kmh/image/upload/v1640050215/image27_frqkzv.png")
st.markdown("### Add a Background Image from a URL")



code = """
import streamlit as st 

def add_bg_from_url():
    st.markdown(
         f'''
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/free-vector/abstract-blue-geometric-shapes-background_1035-17545.jpg?w=2000");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         ''',
         unsafe_allow_html=True
     )

add_bg_from_url() 
"""
st.code(code)



def add_bg_from_url():
    st.markdown(
         f'''
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/free-vector/abstract-blue-geometric-shapes-background_1035-17545.jpg?w=2000");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         ''',
         unsafe_allow_html=True
     )

add_bg_from_url() 





st.markdown("### Add a Background Image from Your Computer")

code = """
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f'''
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    ''',
    unsafe_allow_html=True
    )
add_bg_from_local('My_Image.jpg')    
"""
st.code(code)



import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f'''
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    ''',
    unsafe_allow_html=True
    )
# add_bg_from_local('My_Image.jpg')   


