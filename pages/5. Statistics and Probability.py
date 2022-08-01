# Contents of ~/my_app/streamlit_app.py
import streamlit as st

st.set_page_config(
    page_icon=":blue_heart:",
    layout="wide"
)



st.header("Introduction")
st.write('Statistics refers to the mathematics and techniques with which we understand data. It is a rich, enormous field, more suited to a shelf (or room) in a library than a chapter in a book, and so our discussion will necessarily not be a deep one. Instead, I’ll try to teach you just enough to be dangerous, and pique your interest just enough that you’ll go off and learn more.')
st.markdown('')
st.write('One obvious description of any dataset is simply the data itself:')
code = '''x = [100, 49, 41, 40, 25,
            # ... and lots more
            ]'''

st.code(code, language='python')
st.markdown('As a first approach, you put the friend counts into a histogram using **Counter** and **plt.bar**')
st.markdown('')
st.markdown('Let’s start with this simple list **x** :')

code =  '''
from collections import Counter
import matplotlib.pyplot as plt
import random
random.seed(10) # this ensures we get the same results every time

My_list=[]
for i in range(20):
    My_list.append(random.randint(0,10))
    #My_list = [9, 0, 6, 7, 9, 0, 3, 7, 7, 4, 10, 2, 0, 8, 7, 5, 1, 3, 5, 0]
counter = Counter(My_list)
#Counter({0: 4, 7: 4, 9: 2, 3: 2, 5: 2, 6: 1, 4: 1, 10: 1, 2: 1, 8: 1, 1: 1})
xs = range(max(My_list)) # largest value is 10
ys = [counter[i] for i in xs] 
plt.bar(xs, ys)
plt.axis([min(My_list)-1, max(My_list)+1, min(ys)-1, max(ys)+1])
plt.title("Histogram")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
'''
st.code(code, language='python')


from collections import Counter
import matplotlib.pyplot as plt
import random
random.seed(10) # this ensures we get the same results every time

My_list=[]
for i in range(20):
    My_list.append(random.randint(0,10))
    #My_list = [9, 0, 6, 7, 9, 0, 3, 7, 7, 4, 10, 2, 0, 8, 7, 5, 1, 3, 5, 0]
counter = Counter(My_list)
#Counter({0: 4, 7: 4, 9: 2, 3: 2, 5: 2, 6: 1, 4: 1, 10: 1, 2: 1, 8: 1, 1: 1})
xs = range(max(My_list)) # largest value is 10
ys = [counter[i] for i in xs] 
fig = plt.figure()

plt.axis([min(My_list)-1, max(My_list)+1, min(ys)-1, max(ys)+1])
plt.title("Histogram")
plt.xlabel("X")
plt.ylabel("Y")
plt.bar(xs, ys)
st.pyplot(fig)


st.write('Now try to change the parameters :')


col1, col2= st.columns(2)

n = col1.slider('n = ', 0, 5000,1000)
int_min = col1.number_input("min = ", value=0)
int_max = col1.number_input("max = ", value=400)


randomlist = [range(n+1)]
for i in range(n+1):
    randomlist.append(random.randint(int_min, int_max))


counter = Counter(randomlist)
xs = range(int_max+1)
ys = [counter[i] for i in xs] 

fig= plt.figure()
plt.bar(xs,ys)
col2.pyplot(fig)


