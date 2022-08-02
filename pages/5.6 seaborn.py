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




st.markdown(r"""
##### Scatter Plot
""")
code = '''
import matplotlib.pyplot as plt
sns.scatterplot(x='total_bill', y='tip', data = df_tips,
                 alpha= 0.5, color = 'blue')
plt.xlabel('Total Bill', fontsize = 12, color = 'black')
plt.ylabel('Total Bill', fontsize = 12, color = 'black')
plt.title('seaborn scatterplot', fontsize = 14, color = 'black')
plt.show()
'''
st.code(code, language='python')


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10,7))
sns.scatterplot(x='total_bill', y='tip', data = df_tips, alpha= 0.5,
                 color = 'blue')
plt.xlabel('Total Bill', fontsize = 12, color = 'black')
plt.ylabel('Total Bill', fontsize = 12, color = 'black')
plt.title('seaborn scatterplot', fontsize = 14, color = 'black')
st.pyplot(fig)

st.markdown("In `seaborn`, the `hue` parameter determines which column in the data frame should be used for colour encoding. See example [here](https://datascience.stackexchange.com/questions/46117/meaning-of-hue-in-seaborn-barplot) ")


code = """
sns.scatterplot(x='total_bill', y='tip', data = df_tips,
                            alpha= 0.8, color = 'blue', hue = 'sex')
plt.show()
"""
st.code(code, language='python')

fig = plt.figure(figsize=(10,7))
sns.scatterplot(x='total_bill', y='tip', data = df_tips,
                            alpha= 0.8, color = 'blue', hue = 'sex')
plt.xlabel('Total Bill', fontsize = 12, color = 'black')
plt.ylabel('Total Bill', fontsize = 12, color = 'black')
plt.title('seaborn scatterplot', fontsize = 14, color = 'black')
st.pyplot(fig)


st.markdown("""
We can get the same scatterplot using `sns.relplot` as :
""")

code = """
sns.relplot(x='total_bill', y='tip', data = df_tips,
            alpha= 0.8, color = 'blue', hue = 'sex',
                                                    kind = 'scatter')
plt.show()
"""
st.code(code, language='python')



fig = sns.relplot(x='total_bill', y='tip', data = df_tips,
            alpha= 0.8, color = 'blue', hue = 'sex',
                                                    kind = 'scatter')
plt.xlabel('Total Bill', fontsize = 12, color = 'black')
plt.ylabel('Total Bill', fontsize = 12, color = 'black')
plt.title('seaborn scatterplot', fontsize = 14, color = 'black')
st.pyplot(fig)

st.markdown("""
Then we can easily separate this scatterplot to two (or more) scatterplots 
by (`col=` or `row=`) according to the category of `sex` by example :
""")

code = """
sns.relplot(x='total_bill', y='tip', data = df_tips,
            alpha= 0.8, color = 'blue', hue = 'sex',
                                kind = 'scatter', col = 'sex')
plt.show()
"""
st.code(code, language='python')

fig = sns.relplot(x='total_bill', y='tip', data = df_tips,
            alpha= 0.8, color = 'blue', hue = 'sex',
                                kind = 'scatter', col = 'sex')
st.pyplot(fig)


code = """
sns.relplot(x='total_bill', y='tip', data = df_tips,
            alpha= 0.8, color = 'blue', hue = 'sex',
                                kind = 'scatter', row = 'sex')
plt.show()
"""
st.code(code, language='python')

fig = sns.relplot(x='total_bill', y='tip', data = df_tips,
            alpha= 0.8, color = 'blue', hue = 'sex',
                                kind = 'scatter', row = 'sex')
st.pyplot(fig)





st.markdown("""
We can also use `col=` and `row=` in the same time as :
""")

code = """
sns.relplot(x='total_bill', y='tip', data = df_tips,
            alpha= 0.8, color = 'blue', hue = 'sex',
                                kind = 'scatter', row = 'sex', col = 'smoker')
plt.show()
"""
st.code(code, language='python')



fig = sns.relplot(x='total_bill', y='tip', data = df_tips,
            alpha= 0.8, color = 'blue', hue = 'sex',
                                kind = 'scatter', row = 'sex', col = 'smoker')
st.pyplot(fig)









st.markdown(r"""
##### Bar Plot
""")

code = """
sns.countplot(x='sex', data = df_tips, hue = 'smoker', palette= 'hls')
plt.show()
"""

st.code(code, language='python')

fig = plt.figure(figsize=(10,7))
sns.countplot(x='sex', data = df_tips, hue = 'smoker', palette= 'hls')
st.pyplot(fig)

st.write('More color `palette` [here](https://seaborn.pydata.org/tutorial/color_palettes.html)')



st.markdown(r"""
##### Box Plot
""")

code = """
sns.boxplot(x='sex', y= 'tip', data = df_tips, hue = 'smoker',
                     palette= 'colorblind')
plt.show()
"""

st.code(code, language='python')


fig = plt.figure(figsize=(10,7))
sns.boxplot(x='sex', y= 'tip', data = df_tips, hue = 'smoker', 
            palette= 'colorblind')
st.pyplot(fig)

st.write("""Another methode to get the same plot is by using `catplot` with `kind='box' as :""")

code = """
sns.catplot(x='sex', y= 'tip', data = df_tips, hue = 'smoker', 
                                        palette= 'colorblind', kind='box')
plt.show()
"""

fig = sns.catplot(x='sex', y= 'tip', data = df_tips, hue = 'smoker', 
                                        palette= 'colorblind', kind='box')
st.pyplot(fig)