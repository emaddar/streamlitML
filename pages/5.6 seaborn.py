import matplotlib.pyplot as plt
from pyparsing import srange
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





st.write('We can also use the parameter `size` as :')

code = """
sns.relplot(x='total_bill', y='tip', data = df_tips,
            alpha= 0.8, color = 'blue', hue = 'size',
            kind = 'scatter', row = 'sex', col = 'smoker',
                                                        size = 'size')
plt.show()
"""

st.code(code, language='python')

fig = sns.relplot(x='total_bill', y='tip', data = df_tips,
            alpha= 0.8, color = 'blue', hue = 'size',
            kind = 'scatter', row = 'sex', col = 'smoker',
                                                        size = 'size')
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

st.code(code, language='python')

fig = sns.catplot(x='sex', y= 'tip', data = df_tips, hue = 'smoker', 
                                        palette= 'colorblind', kind='box')
st.pyplot(fig)






st.markdown(r"""
##### Histograme
""")

code = """
sns.histplot(x= 'tip', data = df_tips, bins = 25)
plt.show()
"""

st.code(code, language='python')


fig = plt.figure(figsize = (10,7))
sns.histplot(x= 'tip', data = df_tips, bins = 25)
st.pyplot(fig)



st.markdown(r"""
###### option : kde = True :
compute a **kernel density estimate** to smooth the distribution and show on the plot as (one or more) line(s). Only relevant with univariate data.

See more in [Visualizing distributions of data](https://seaborn.pydata.org/tutorial/distributions.html).
""")


code = """
sns.histplot(x = 'total_bill', data = df_tips, hue = 'sex', kde = True) 
"""
st.code(code, language='python')
fig = plt.figure(figsize=(10,7))
sns.histplot(x = 'total_bill', data = df_tips, hue = 'sex', kde = True) 
st.pyplot(fig)



st.markdown(r"""
###### option : multiple="stack" :
emphasizes the part-whole relationship between the variables
""")

code = """
sns.histplot(x = 'total_bill', data = df_tips, hue = 'sex', kde = True,
                                                                    multiple="stack") 
"""
st.code(code, language='python')
fig = plt.figure(figsize=(10,7))
sns.histplot(x = 'total_bill', data = df_tips, hue = 'sex', kde = True, multiple="stack") 
st.pyplot(fig)






st.markdown(r"""
###### option : multiple="dodge" :
 moves them horizontally and reduces their width.
""")

code = """
sns.histplot(x = 'total_bill', data = df_tips, hue = 'sex', kde = True,
                                                                    multiple="dodge") 
"""
st.code(code, language='python')
fig = plt.figure(figsize=(10,7))
sns.histplot(x = 'total_bill', data = df_tips, hue = 'sex', kde = True, multiple="dodge") 
st.pyplot(fig)




st.markdown(r"""
###### Note :
We can **not** use the options `kind = 'hist'` in `relplot`. But we can always use `plt.subplots`.
""")

code = """
fig, ax = plt.subplots(1,2,figsize=(10,7))
plt.sca(ax[0])
sns.histplot(x= 'tip', data = df_tips[df_tips['smoker']=='No'],
                         bins = 25, color = 'red')
plt.title("Smoker = No")

plt.sca(ax[1])
sns.histplot(x= 'tip', data = df_tips[df_tips['smoker']=='Yes'], bins = 25)
plt.title("Smoker = Yes")

plt.show()
"""

st.code(code, language='python')

fig, ax = plt.subplots(1,2,figsize=(10,3))
plt.sca(ax[0])
sns.histplot(x= 'tip', data = df_tips[df_tips['smoker']=='No'],
                         bins = 25, color = 'red')
plt.title("Smoker = No")

plt.sca(ax[1])
sns.histplot(x= 'tip', data = df_tips[df_tips['smoker']=='Yes'], bins = 25)
plt.title("Smoker = Yes")

st.pyplot(fig)


st.markdown(r"""
###### Note :
In case we have only one row in `subplots` as in our example, we can't use `ax[0,0]` for first plot ...
""")

















for i in range(5):
    st.markdown("")

st.markdown(r"""
#### Line Plot
""")


st.markdown(r"""
##### Load and clean data
""")
code = """
import pandas as pd
df_air = pd.read_csv('https://raw.githubusercontent.com/ishanag9/air-quality-prediction/master/AirQualityUCI.csv')
print(df_air.head())
"""

st.code(code, language='python')

import pandas as pd
df_air = pd.read_csv('https://raw.githubusercontent.com/ishanag9/air-quality-prediction/master/AirQualityUCI.csv')
st.dataframe(df_air.head())


st.write('[Understanding DataSet](https://github.com/ishanag9/air-quality-prediction/blob/master/README.md)')




st.markdown(r"""
###### Drop `Na` from `Time` column
""")
code = """
df_air = df_air.dropna(subset=['Time'], axis = 0)
"""
st.code(code, language='python')

df_air = df_air.dropna(subset=['Time'], axis = 0)




code= """
sns.lineplot(x= 'Time', y = 'NOx(GT)', data = df_air)
plt.xticks(rotation = 90)
plt.show()
"""

st.code(code, language='python')

fig = plt.figure(figsize=(10,7))
sns.lineplot(x= 'Time', y = 'NOx(GT)', data = df_air)
plt.xticks(rotation = 90)
st.pyplot(fig)







for i in range(5):
    st.markdown("")

st.markdown(r"""
#### Pairplot
[seaborn.pairplot](https://seaborn.pydata.org/generated/seaborn.pairplot.html) 
Plot pairwise relationships in a dataset.

By default, this function will create a grid of Axes such that each numeric variable in data will by shared across the y-axes across a single row and the x-axes across a single column. 
""")


st.markdown(r"""
##### Load data
""")
code = """
penguins = sns.load_dataset("penguins")
print(penguins.head())
"""

st.code(code, language='python')


penguins = sns.load_dataset("penguins")
st.dataframe(penguins.head())



code = """
sns.pairplot(penguins, hue="species")
plt.show()
"""

st.code(code, language='python')


fig= sns.pairplot(penguins, hue="species")
st.pyplot(fig)
 