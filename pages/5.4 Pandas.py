# Contents of ~/my_app/streamlit_app.py
from random import randint
from re import L
from matplotlib.pyplot import hist, show
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_icon=":blue_heart:"
)


st.markdown("<h1 style='text-align: center; color: black;'>Pandas</h1>", unsafe_allow_html=True)


st.markdown(r"""
[pandas](https://en.wikipedia.org/wiki/Pandas_(software)) is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series.
Its name is a play on the phrase "Python data analysis" itself. It is developed by the  American software developer and businessman [Wes McKinney](https://en.wikipedia.org/wiki/Wes_McKinney)

##### Installing with Miniconda
In terminal :
""")
st.code("conda install pandas", language='python')

st.markdown(r"""
##### Installing from PyPI
In terminal :
""")

st.code("pip install pandas", language='python')

st.markdown(r"""
##### To install a specific pandas version:
In terminal :
""")

st.code("conda install pandas=0.20.3", language='python')

st.markdown(r"""
##### Check pandas version
""")
code = """
import pandas as pd
print(pd.__version__)
"""
st.code(code, language='python')


st.markdown(r"""
##### Read CSV File
""")
code = """
df = pd.read_csv(r'c:\~\data.csv')  # df is dataframe
"""
st.code(code, language='python')




st.markdown(r"""
##### Get the dimensionality of the DataFrame
""")
code = """
df = pd.read_csv('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')
print(df.shape)  # (150, 5) : 105 rows and 5 columns
"""
st.code(code, language='python')

import pandas as pd
df = pd.read_csv('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')
st.dataframe(df)

st.markdown(r"""
##### Get the first n rows for the object based on position.
""")
code = """
print(df.head(5))  # first 5 rows
"""
st.code(code, language='python')


st.markdown(r"""
##### Get the last n rows
""")
code = """
print(df.tail(3))  # first 5 rows
"""
st.code(code, language='python')


st.markdown(r"""
##### Print a concise summary of a DataFrame.
""")
code = """
df.info()
"""
st.code(code, language='python')

st.write('This method prints information about a DataFrame including the index dtype and columns, non-null values and memory usage.')




st.markdown(r"""
##### Generate descriptive statistics.
""")
code = """
print(df.describe())
"""
st.code(code, language='python')

st.write('Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.')




st.markdown(r"""
##### Check if any value is NaN in a Pandas DataFrame
""")
code = """
print(df.isna().any())
"""
st.code(code, language='python')

st.write('Or')

code = """
print(df.isnull().any())
"""
st.code(code, language='python')

st.write("Both ```isna()``` and ```isnull()``` functions are used to find the missing values in the pandas dataframe. ```isnull()``` and ```isna()``` literally does the same things.")

st.write("We can add ```.sum()``` to get the sum of missing values in all our DataFrame")

code = """
print(df.isnull().any().sum())
"""

st.code(code, language='python')




st.markdown(r"""
##### Get column labels of the DataFrame
""")
code = """
print(list(df.columns))
"""
st.code(code, language='python')




st.markdown(r"""
##### Sort dataframe by columns
""")
code = """
df = df.sort_values(by=['petal.length', 'petal.width'], axis=0, ascending=True)  # ascending = True by default
"""
st.code(code, language='python')





st.markdown(r"""
##### Get unique values of column
""")
code = """
print(df['variety'].unique())  # ['Setosa' 'Versicolor' 'Virginica']
"""
st.code(code, language='python')



st.markdown(r"""
##### Get counts of unique values
""")
code = """
print(df['variety'].value_counts())  # Setosa        50, Versicolor    50, Virginica     50
"""
st.code(code, language='python')

st.write('We can also use the option ```normalize = True``` to the relative frequencies of the unique values.')
code = """
print(df['variety'].value_counts(normalize=True, ascending=False))  # ascending=True by default
"""
st.code(code, language='python')


 

 
st.markdown(r"""
##### Select a subset of a DataFrame
###### By column(s)

""")
code = """
df1 = df[['sepal.length', 'sepal.width']]
print(df1.head(5))
"""
st.code(code, language='python')

df1 = df[['sepal.length', 'sepal.width']]
st.dataframe(df1.head(5))


st.markdown(r"""
###### Select rows based on a conditional expression

""")
code = """
df2 = df[df['sepal.length'] > 4.0]
print(df2)
"""
df2 = df[df['sepal.length'] > 4.0]
st.code(code, language='python')
st.dataframe(df2)

code = """
df3 = df[(df["sepal.length"] > 4) | (df["sepal.width"] <= 3)]
print(df3)
"""
st.code(code, language='python')
df3 = df[(df["sepal.length"] > 4) | (df["sepal.width"] <= 3)]
st.dataframe(df3)


code = """
df3 = df[(df["sepal.length"] > 4) & (df["sepal.width"] <= 3)]
print(df3)
"""
st.code(code, language='python')
df4 = df[(df["sepal.length"] > 4) & (df["sepal.width"] <= 3)]
st.dataframe(df4)



st.markdown(r"""
###### I’m interested in rows 10 till 25 and columns 3 to 5

""")
code = """
print(df.iloc[9:25, 2:5])
"""
st.code(code, language='python')
df = pd.read_csv('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')
st.dataframe(df.iloc[9:25, 2:5])

st.markdown(r"""
###### isin
Similar to the conditional expression, the ```isin()``` conditional function returns a True for each row the values are in the provided list. 
""")
code = '''
print(df[df['variety'].isin(['Setosa', 'Virginica'])])
'''
st.code(code, language='python')
st.dataframe(df[df['variety'].isin(['Setosa', 'Virginica'])])

st.write('for more information about selecting a subset of a DataFrame, [click here](https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html).')


for i in range(4):
    st.markdown('')


st.markdown(r"""
#### groupby


""")
code = """
print(   df.groupby('variety')['sepal.width'].count()  )
"""
st.code(code, language='python')

st.dataframe(df.groupby('variety')['sepal.width'].count())


code = """
print(   df.groupby('variety')['sepal.width'].sum()  )
"""
st.code(code, language='python')

st.dataframe(df.groupby('variety')['sepal.width'].sum())

st.write('And we can put ```count()```, ```sum()```, ```mean()``` together using ```agg``` function as :')


code = """
print(   df.groupby('variety')['sepal.width'].agg (['sum', 'count'])  )
"""
st.code(code, language='python')

st.dataframe(df.groupby('variety')['sepal.width'].agg (['sum', 'count', 'mean']))


st.write('Using python dictionnaire :')

code = """
print(   (df.groupby('variety')['sepal.width'].agg ({'sum', 'count', 'mean', 'max', 'min'}))  )
"""
st.code(code, language='python')

st.dataframe(df.groupby('variety')['sepal.width'].agg ({'sum', 'count', 'mean', 'max', 'min'}))


st.write('We can also use ```numpy``` functions as :')

code = """
import numpy as np
print(   (df.groupby('variety')['sepal.width'].agg ([np.min, np.max, np.mean, np.std, np.var]))  )
"""
st.code(code, language='python')
import numpy as np
st.dataframe(   (df.groupby('variety')['sepal.width'].agg ([np.min, np.max, np.mean, np.std, np.var]))  )







st.markdown(r"""
#### pivot_table
Create a spreadsheet-style pivot table as a DataFrame 
""")
code = """
df4 =  df.pivot_table(index='variety',  values='sepal.width', aggfunc=['sum', 'count', 'mean', 'max', 'min'])
print(df4)
"""
st.code(code, language='python')
df4 =  df.pivot_table(index='variety',  values='sepal.width', aggfunc=['sum', 'count', 'mean', 'max', 'min'])
st.dataframe(df4)

st.write("Another example :")
code = """
titanic = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

titanic_pivot =  titanic.pivot_table(index='Sex',  columns='Survived', values='Fare', aggfunc=['sum', 'count', 'mean', 'max', 'min'])
print(titanic_pivot)
"""
st.code(code, language='python')
titanic = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

titanic_pivot =  titanic.pivot_table(index='Sex',  columns='Survived', values='Fare', aggfunc=['sum', 'count', 'mean', 'max', 'min'])
st.dataframe(titanic_pivot)

st.write("Both ```pivot_table``` and ```groupby``` are used to aggregate your dataframe. The difference is only with regard to the shape of the result.")

st.markdown(r"""
#### Add A New Column To An Existing Pandas DataFrame
we can do for example : 
""")
code = """
df['My_calculations'] = df['sepal.length']/ df['sepal.width']
print(df.head())
"""
st.code(code, language='python')
df['My_calculations'] = df['sepal.length']/ df['sepal.width']
st.dataframe(df.head())



st.markdown(r"""
#### Visualisation
Create a spreadsheet-style pivot table as a DataFrame 
""")

code = """
df['sepal.length'].hist(bins=12, xrot=25)
"""
st.code(code, language='python')
fig = plt.figure()
df['sepal.length'].hist(bins=12, xrot=25)
st.pyplot(fig)


st.write('Or :')

code = """
df.hist(column='sepal.length', bins=12, xrot=25)
"""
st.code(code, language='python')

fig = plt.figure()
df.hist(column='sepal.length', bins=12, xrot=25)
st.pyplot(fig)

st.write('Or :')

code = """
df['sepal.length'].plot(kind = 'hist')
"""
st.code(code, language='python')


fig = plt.figure()
df['sepal.length'].plot(kind = 'hist')
st.pyplot(fig)

st.write('Another kinds of plots :')

code = """
df['sepal.length'].plot(kind = 'box')
df['sepal.length'].plot(kind = 'density')
"""
st.code(code, language='python')

fig = plt.figure()

ax1 = fig.add_subplot(1,2,1)
df['sepal.length'].plot(kind = 'box')

ax2 = fig.add_subplot(1,2,2)
df['sepal.length'].plot(kind = 'density')

st.pyplot(fig)


st.write('Click [here](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html) for more plot kinds')




st.markdown(r"""
#### Missing values

`missingno` is an excellent and simple to use Python library that provides a series of visualisations to understand the presence and distribution of missing data within a pandas dataframe. 
""")
code = '''
df_house = pd.read_csv('https://raw.githubusercontent.com/AGOOR97/Hands-on_Machine_Learning_Topics/main/datasets/housing/housing.csv')
df_house.head()
'''
st.code(code, language='python')
df_house = pd.read_csv('https://raw.githubusercontent.com/AGOOR97/Hands-on_Machine_Learning_Topics/main/datasets/housing/housing.csv')
st.dataframe(df_house.head())


code = '''
import missingno as msno
msno.bar(df,  color="dodgerblue");
'''
st.code(code, language='python')
import missingno as msno

import matplotlib.pyplot as plt
fig = plt.figure()
msno.bar(df_house,  color="dodgerblue");
st.pyplot(fig)


code = '''
msno.matrix(df_house);
'''
st.code(code, language='python')
fig = plt.figure()
msno.matrix(df_house);
st.pyplot(fig)


st.write("Get columns with `null`s")
code = """
df_house.isnul().any()
"""
st.code(code, language='python')

st.dataframe(df_house.isna().any())

st.write("Get sum of `null`values in each columns")
code = """
df_house.isnul().sum()
"""
st.code(code, language='python')

st.dataframe(df_house.isna().sum())
st.write("Then we can plot it as :")
code = """
df_house.isnul().sum().plot()
df_house.isna().sum().plot(kind = 'barh')
"""
st.code(code, language='python')
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
df_house.isna().sum().plot()
ax1 = fig.add_subplot(1,2,2)
df_house.isna().sum().plot(kind = 'barh')
st.pyplot(fig)



st.write("Get only `null`values from `total_bedrooms`column :")
code = """
print(df_house[df_house['total_bedrooms'].isna()])
"""
st.code(code, language='python')

st.dataframe(df_house[df_house['total_bedrooms'].isna()])


st.markdown(r"""
##### dropna

Remove missing values
""")
code = '''
df_house.dropna(subset='total_bedrooms', axis=0)
'''
st.code(code, language='python')

st.write('''
* By default `subset=False`, in this case, `dropna` will frop all `NaN` in this DataFrame.
* Use the option `inplace = True` to remove `Nan` from `df_house` DataFrame.
''')





st.markdown(r"""
##### fillna

Fill NA/NaN values using the specified method.
""")
code = '''
df_house.fillna(df_house['total_bedrooms'].mean())
'''
st.code(code, language='python')
st.write('This function will fill all `Nan` in `total_bedrooms` column by the `mean` of this column ')





st.markdown(r"""
#### Create dataframe
""")
code = '''
My_df = pd.DataFrame({'Name' = ['Name1', 'Name2', 'Name3'],
                    'Age' = [18,19,np.nan],
                    'Job' = ['Job1', 'Job2', 'Job3']
                    })
print(My_df)
'''
st.code(code, language='python')

My_df = pd.DataFrame({'Name' : ['Name1', 'Name2', 'Name3'],
                    'Age' : [18,19,np.nan],
                    'Job' : ['Job1', 'Job2', 'Job3']
                    })
st.dataframe(My_df)



st.markdown(r"""
#### Savz DataFrame as csv ...:
""")
code = '''
My_df.to_csv(r'c:\~\MyData.csv')
'''
st.code(code, language='python')

