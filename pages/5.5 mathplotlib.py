from turtle import color
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_icon=":blue_heart:"
)


st.markdown("<h1 style='text-align: center; color: black;'>mathplotlib</h1>", unsafe_allow_html=True)


st.markdown(r"""
[mathplotlib](https://matplotlib.org/) is a comprehensive library for creating static, animated, and interactive visualizations in Python. 

##### Installing from PyPI
In terminal :
""")
st.code("pip install mathplotlib", language='python')

st.markdown(r"""
##### Import mathplotlib
""")
st.code("import matplotlib.pyplot as plt", language='python')



st.markdown(r"""
##### Read data
""")
code = """
import pandas as pd
df_house = pd.read_csv('https://raw.githubusercontent.com/AGOOR97/Hands-on_Machine_Learning_Topics/main/datasets/housing/housing.csv')
df_house.head()
"""
st.code(code, language='python')

import pandas as pd
df_house = pd.read_csv('https://raw.githubusercontent.com/AGOOR97/Hands-on_Machine_Learning_Topics/main/datasets/housing/housing.csv')
st.dataframe(df_house.head())



st.markdown(r"""
### Scatter plot
###### Example 1:
""")

code = """
import numpy as np
x = np.linspace(10,50,60)
y = np.exp(x)
plt.scatter(x,y)
plt.show()
"""
st.code(code, language='python')
import numpy as np
x = np.linspace(10,50,60)
y = np.exp(x)
fig = plt.figure(figsize=(10,6))
plt.scatter(x,y)
st.pyplot(fig)


st.markdown(r"""
###### Example 2:
""")

code = """
x = 2* np.random.rand(100,1)
y = 4+3*x+np.random.rand(100,1)
z = 4-3*x+np.random.rand(100,1)

plt.figure(figsize=(10,6))
plt.scatter(x,y, color = 'red', label = "x vs. y")
plt.scatter(x,z, color= 'green', label = 'x vs. z')
plt.xlabel('x', fontsize = 12)
plt.title('Scatter plot from some test data', fontsize = 14, color = 'black')
plt.legend()
plt.show()
"""
st.code(code, language='python')

x = 2* np.random.rand(100,1)
y = 4+3*x+np.random.rand(100,1)
z = 4-3*x+np.random.rand(100,1)

fig = plt.figure(figsize=(10,6))
plt.scatter(x,y, color = 'red', label = "x vs. y")
plt.scatter(x,z, color= 'green', label = 'x vs. z')
plt.xlabel('x', fontsize = 12)
plt.title('Scatter plot from some test data', fontsize = 14, color = 'black')
plt.legend()
st.pyplot(fig)

st.markdown(r"""
###### Example 3:
""")



code = """
plt.figure(figsize=(10,6))
plt.scatter(df_house['longitude'], df_house['latitude'], alpha=0.4, color= 'green', marker='o', s = 30)
plt.xlabel('Longitude', fontsize = 14)
plt.ylabel('Latitude', fontsize = 14)
plt.title('Log. vs Lat. for Housing Dataset', fontsize = 14, color = 'red')
plt.show()
"""
st.code(code, language='python')

fig = plt.figure(figsize=(10,6))
plt.scatter(df_house['longitude'], df_house['latitude'], alpha=0.2, color= 'blue', marker='o', s = 30)
plt.xlabel('Longitude', fontsize = 14)
plt.ylabel('Latitude', fontsize = 14)
plt.title('Log. vs Lat. for Housing Dataset', fontsize = 14, color = 'red')
st.pyplot(fig)






st.write('''Matplotlib allows you to adjust the **transparency** of a graph plot using the `alpha` attribute.
 If you want to make the graph plot more transparent, then you can make alpha less than 1, such as 0.5 or 0.25. If you want to make the graph plot less transparent, then you can make alpha greater than 1.''')


st.write('')
st.markdown(r"""
**Now we can make the siez of markers depende on another variable like population, and also we can give some colors dependinfg the houses prices by example :** 
""")



code = """
plt.figure(figsize=(10,6))
sc = plt.scatter(df_house['longitude'], df_house['latitude'], alpha=0.2,
                      marker='o',
                     c= df_house['median_house_value'],
                       s = df_house['population']/50,
                       cmap=plt.get_cmap('jet'),
                       label = "Population")
plt.colorbar(sc)

plt.xlabel('Longitude', fontsize = 14)
plt.ylabel('Latitude', fontsize = 14)
plt.title('Log. vs Lat. for Housing Dataset', fontsize = 14, color = 'red')
plt.legend(loc = "upper right")
plt.show()
"""
st.code(code, language='python')

fig = plt.figure(figsize=(10,6))
sc = plt.scatter(df_house['longitude'], df_house['latitude'], alpha=0.2,
                      marker='o',
                     c= df_house['median_house_value'],
                       s = df_house['population']/50,
                       cmap=plt.get_cmap('jet'),
                       label = "Population")
plt.colorbar(sc)

plt.xlabel('Longitude', fontsize = 14)
plt.ylabel('Latitude', fontsize = 14)
plt.title('Log. vs Lat. for Housing Dataset', fontsize = 14, color = 'red')
plt.legend(loc = "upper right")
st.pyplot(fig)

st.write('`cmap` is referred to [color map](https://matplotlib.org/stable/tutorials/colors/colormaps.html)')




st.write('')
st.write('')
st.write('')

st.markdown(r"""
### Try it yourself
""")

col1, col2 = st.columns(2)

from matplotlib.lines import Line2D


x = col1.selectbox('x = ',df_house.columns.values)
y = col1.selectbox('y = ',df_house.columns.values)
s = col1.selectbox('Size = ',df_house.columns.values)
c = col1.selectbox('Color = ',df_house.columns.values)
marker = col1.selectbox('Marker = ',Line2D.markers)
alpha = col1.slider('Transparency', 0.0, 1.0, 0.3)
cmap = st.selectbox('Select color map :', ('Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'))
ft = col1.slider('Font size', 1, 30, 15)
fc = col1.color_picker('Select font color', '#2300F9')
Title = col1.text_input('Input title : ', 'Log. vs Lat. for Housing Dataset')
legend_location = col1.selectbox('Location = ',('best', 'upper right', 'upper left', 'lower right', 'lower left', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'))

fig = plt.figure(figsize=(10,20))
sc = plt.scatter(df_house[x], df_house[y], alpha=alpha,
                      marker=marker,
                     c= df_house['median_house_value'],
                       s = df_house['population']/50,
                       cmap=plt.get_cmap(cmap),
                       label = s)
plt.colorbar(sc)

plt.xlabel(x, fontsize = ft, color = fc)
plt.ylabel(y, fontsize = ft, color = fc)
plt.title(Title, fontsize = ft, color = fc)
plt.legend(loc = legend_location)
col2.pyplot(fig)

st.markdown(r"""
##### Note that we can get almost the same results using `pandas` visualisation as :
""")

code = """
df_house.plot(x= 'longitude', y= 'latitude',
            kind = 'scatter', 
            s = df_house['population']/50, 
            color = df_house['median_house_value'],
            cmap = plt.get_cmap('jet'),
            figsize = (14,8),
            xlabel = 'longitude',
            ylabel = 'latitude',
            title = 'Log. vs Lat. for Housing Dataset',
            marker = '^'
)
"""
st.code(code, language= 'python')





st.markdown(r"""
### LinePlot
###### Example 1:
""")

code = """
# creat a test dataset
x = np.linspace(1,5,100)
y = np.exp(x)
z = np.log2(x)

fig, ax = plt.subplots()
ax.plot(x,y, marker='^', label = 'exp(x)',  color = 'blue', markersize = 5)
ax.set_xlabel('x')    # !!! No xlabel fonction here
ax.set_ylabel('y', rotation=0, color = 'r', fontsize = 25)
ax.legend(loc = 'upper left') 

ax2 = ax.twinx()
ax2.plot(x,z, marker= 'o', label = 'log(x)' , color = 'green', markersize = 5)
ax2.set_ylabel('z', rotation=0, color = 'r', fontsize = 25)
ax2.legend(loc = 'upper right')

plt.title("LinePLot with Axes twins")

plt.show()
"""
st.code(code, language='python')

x = np.linspace(1,5,100)
y = np.exp(x)
z = np.log2(x)

fig, ax = plt.subplots(figsize = (12,10))
ax.plot(x,y, marker='^', label = 'exp(x)',  color = 'blue', markersize = 5)
ax.set_xlabel('x')    # !!! No xlabel fonction here
ax.set_ylabel('y', rotation=0, color = 'r', fontsize = 25)
ax.legend(loc = 'upper left') 

ax2 = ax.twinx()
ax2.plot(x,z, marker= 'o', label = 'log(x)' , color = 'green', markersize = 5)
ax2.set_ylabel('z', rotation=0, color = 'r', fontsize = 25)
ax2.legend(loc = 'upper right')

plt.title("LinePLot with Axes twins")
st.pyplot(fig)

st.write('`Axes.twinx` : create a twin Axes sharing the xaxis.')
st.write('In `Axes` case, we use : `ax.set_xlabel` and `ax.set_ylabel` in place of `xlabels` and `ylabel`.')





st.markdown(r"""
### Histogram
###### Example 1:
""")

code = """
plt.figure(figsize = (12,10))
plt.hist(df_house['median_income'], 
            bins = 10 , 
            color='#7E6CEA')
plt.xlabel('Median income', fontsize= 12, color = 'black')
plt.ylabel('Frequency', fontsize= 12, color = 'black')
plt.title('plt Histogram', fontsize = 14, color = 'black')
plt.show()
"""
st.code(code, language='python')


fig = plt.figure(figsize = (12,10))
plt.hist(df_house['median_income'], 
            bins = 10 , 
            color='#7E6CEA')
plt.xlabel('Median income', fontsize= 12, color = 'black')
plt.ylabel('Frequency', fontsize= 12, color = 'black')
plt.title('plt Histogram', fontsize = 14, color = 'black')
st.pyplot(fig)



st.markdown(r"""
###### Example 2:
""")

code ="""
plt.figure(figsize = (7,5))
plt.hist(df_house['median_house_value'], 
            bins = 10 , 
            color='red', 
            histtype='step') # histtype take one of these values : {'bar', 'barstacked', 'step', 'stepfilled'}
plt.xlabel('Median house value', fontsize= 12, color = 'black')
plt.ylabel('Frequency', fontsize= 12, color = 'black')
plt.title('plt Histogram with histtype= step', fontsize = 14, color = 'black')
plt.show()
"""
st.code(code, language='python')


fig = plt.figure(figsize = (7,5))
plt.hist(df_house['median_house_value'], 
            bins = 10 , 
            color='red', 
            histtype='step')
plt.xlabel('Median house value', fontsize= 12, color = 'black')
plt.ylabel('Frequency', fontsize= 12, color = 'black')
plt.title('plt Histogram with histtype= step', fontsize = 14, color = 'black')
st.pyplot(fig)








st.markdown(r"""
### BoxPlot
""")

code = """
plt.figure(figsize = (7,5))
plt.boxplot(df_house['median_house_value'] 
            )
plt.xlabel('Median house value', fontsize= 12, color = 'black')
plt.ylabel('Values', fontsize= 12, color = 'black')
plt.title('plt BoxPlot', fontsize = 14, color = 'black')
plt.show()
"""
st.code(code, language='python')


fig = plt.figure(figsize = (7,5))
plt.boxplot(df_house['median_house_value'] )
plt.xlabel('Median house value', fontsize= 12, color = 'black')
plt.ylabel('Values', fontsize= 12, color = 'black')
plt.title('plt BoxPlot', fontsize = 14, color = 'black')
st.pyplot(fig)

















st.markdown(r"""
### BarPlot
""")

code = """
var = df_house['ocean_proximity'].value_counts()


plt.figure(figsize = (7,5))
plt.bar(x = var.index , height = var.values)
plt.xlabel('Median house value', fontsize= 12, color = 'black')
plt.ylabel('Values', fontsize= 12, color = 'black')
plt.title('plt BarPlot', fontsize = 14, color = 'black')
plt.show()
"""
st.code(code, language='python')

var = df_house['ocean_proximity'].value_counts()

fig = plt.figure(figsize = (7,5))
plt.bar(x = var.index , height = var.values)
plt.xlabel('Median house value', fontsize= 12, color = 'black')
plt.ylabel('Values', fontsize= 12, color = 'black')
plt.title('plt BarPlot', fontsize = 14, color = 'black')
st.pyplot(fig)








st.markdown(r"""
### Pie chart
""")

code = """
var = df_house['ocean_proximity'].value_counts()
plt.figure(figsize = (7,5))
plt.pie(x = var.values , labels= var.index, 
        autopct= '%1.2f%%',
        explode=[0.1 for _ in range(len(var.values))]) 
plt.xlabel('Median house value', fontsize= 12)
plt.ylabel('Values', fontsize= 12)
plt.title('plt BarPlot', fontsize = 14)
plt.show()
"""
st.code(code, language='python')

st.write("""`autopct` enables you to display the **percent value** using Python string formatting.
`explode` specifies the fraction of the radius with which to offset each wedge.
""")


var = df_house['ocean_proximity'].value_counts()
fig = plt.figure(figsize = (7,5))
plt.pie(x = var.values , labels= var.index, 
        autopct= '%1.2f%%',
        explode=[0.1 for _ in range(len(var.values))]) 
plt.xlabel('Median house value', fontsize= 12)
plt.ylabel('Values', fontsize= 12)
plt.title('plt BarPlot', fontsize = 14)
st.pyplot(fig)








st.markdown(r"""
### Subplot
""")



def my_hist(df, bins, colors, fontsize, fontcolor, xlabel, ylabel, title):
    plt.hist(df, 
                bins = bins , 
                color = colors)
    plt.xlabel(xlabel, fontsize= fontsize, color = fontcolor)
    plt.ylabel(ylabel, fontsize= fontsize, color = fontcolor)
    plt.title(title, fontsize = fontsize, color = fontcolor)
    return fig


list_val = df_house['ocean_proximity'].value_counts().index.tolist()
val = df_house['ocean_proximity'].value_counts().values

fig, ax = plt.subplots(2,2, figsize=(12,10)) # 2 rows, 2 colons
fig.tight_layout(pad=5)


plt.sca(ax[0,0])
to_plot = df_house[df_house['ocean_proximity']==list_val[0]]['median_income']
fig = my_hist(df = to_plot, bins = 50, colors= 'blue',
                fontsize = 10, fontcolor= 'black',
                xlabel='Median income',
                ylabel='Frequency',
                title=list_val[0])

plt.sca(ax[0,1])
to_plot = df_house[df_house['ocean_proximity']==list_val[1]]['median_income']
fig = my_hist(df = to_plot, bins = 50, colors= 'blue',
                fontsize = 10, fontcolor= 'black',
                xlabel='Median income',
                ylabel='Frequency',
                title=list_val[1])
plt.sca(ax[1,0])
to_plot = df_house[df_house['ocean_proximity']==list_val[2]]['median_income']
fig = my_hist(df = to_plot, bins = 50, colors= 'blue',
                fontsize = 10, fontcolor= 'black',
                xlabel='Median income',
                ylabel='Frequency',
                title=list_val[2])
plt.sca(ax[1,1])
to_plot = df_house[df_house['ocean_proximity']==list_val[3]]['median_income']
fig = my_hist(df = to_plot, bins = 50, colors= 'blue',
                fontsize = 10, fontcolor= 'black',
                xlabel='Median income',
                ylabel='Frequency',
                title=list_val[3])

st.pyplot(fig)


for i in range(5):
    st.write("")


st.markdown(r"""
To plot this figure 👆, I have two methods, but first lets define the next function :
""")

code ="""
def my_hist(df, bins, colors, fontsize, fontcolor, xlabel, ylabel, title):
    plt.hist(df, 
                bins = bins , 
                color = colors)
    plt.xlabel(xlabel, fontsize= fontsize, color = fontcolor)
    plt.ylabel(ylabel, fontsize= fontsize, color = fontcolor)
    plt.title(title, fontsize = fontsize, color = fontcolor)
    return fig
"""
st.code(code, language='python')



st.markdown(r"""
#### Solution 1
""")

code = """

list_val = df_house['ocean_proximity'].value_counts().index.tolist()

fig, ax = plt.subplots(2,2, figsize=(12,10)) # 2 rows, 2 colons
fig.tight_layout(pad=5)


plt.sca(ax[0,0])
to_plot = df_house[df_house['ocean_proximity']==list_val[0]]['median_income']
fig = my_hist(df = to_plot, bins = 50, colors= 'blue',
                fontsize = 10, fontcolor= 'black',
                xlabel='Median income',
                ylabel='Frequency',
                title=list_val[0])

plt.sca(ax[0,1])
to_plot = df_house[df_house['ocean_proximity']==list_val[1]]['median_income']
fig = my_hist(df = to_plot, bins = 50, colors= 'blue',
                fontsize = 10, fontcolor= 'black',
                xlabel='Median income',
                ylabel='Frequency',
                title=list_val[1])
plt.sca(ax[1,0])
to_plot = df_house[df_house['ocean_proximity']==list_val[2]]['median_income']
fig = my_hist(df = to_plot, bins = 50, colors= 'blue',
                fontsize = 10, fontcolor= 'black',
                xlabel='Median income',
                ylabel='Frequency',
                title=list_val[2])
plt.sca(ax[1,1])
to_plot = df_house[df_house['ocean_proximity']==list_val[3]]['median_income']
fig = my_hist(df = to_plot, bins = 50, colors= 'blue',
                fontsize = 10, fontcolor= 'black',
                xlabel='Median income',
                ylabel='Frequency',
                title=list_val[3])

plt.show()
"""

st.code(code, language='python')
st.markdown("""
* Use `tight-layout` to fit plots within your figure cleanly.
* `plt.sca` set the current Axes to ax and the current Figure to the parent of ax.
""")

st.markdown(r"""
#### Solution 2
""")



code = """
list_val = df_house['ocean_proximity'].value_counts().index[:4].tolist()
list_val = np.array(list_val).reshape(2,2)

fig, ax = plt.subplots(2,2, figsize=(12,10)) # 2 rows, 2 colons
fig.tight_layout(pad=5)

for i in range(2):
    for j in range(2):

        plt.sca(ax[i,j])
        to_plot = df_house[df_house['ocean_proximity']==list_val[i,j]]['median_income']
        fig = my_hist(df = to_plot, bins = 50, colors= 'blue',
                        fontsize = 10, fontcolor= 'black',
                        xlabel='Median income',
                        ylabel='Frequency',
                        title=list_val[i,j])
plt.show()
"""
st.code(code, language='python')



st.write('Lets now create a funtion to generate  random `HEX`number :')

code = """
import random
def my_hex():
    r = lambda: random.randint(0,255)
    return '#%02X%02X%02X' % (r(),r(),r())
"""

st.code(code, language='python')


code = """
list_val = df_house['ocean_proximity'].value_counts().index[:4].tolist()
list_val = np.array(list_val).reshape(2,2)

fig, ax = plt.subplots(2,2, figsize=(12,10)) # 2 rows, 2 colons
fig.tight_layout(pad=5)

for i in range(2):
    for j in range(2):

        plt.sca(ax[i,j])
        to_plot = df_house[df_house['ocean_proximity']==list_val[i,j]]['median_income']
        fig = my_hist(df = to_plot, bins = 50, colors= my_hex(),
                        fontsize = 10, fontcolor= my_hex(),
                        xlabel='Median income',
                        ylabel='Frequency',
                        title=list_val[i,j])
plt.show()
"""

st.code(code, language='python')


import random
def my_hex():
    r = lambda: random.randint(0,255)
    return '#%02X%02X%02X' % (r(),r(),r())


list_val = df_house['ocean_proximity'].value_counts().index[:4].tolist()
list_val = np.array(list_val).reshape(2,2)

fig, ax = plt.subplots(2,2, figsize=(12,10)) # 2 rows, 2 colons
fig.tight_layout(pad=5)

for i in range(2):
    for j in range(2):

        plt.sca(ax[i,j])
        to_plot = df_house[df_house['ocean_proximity']==list_val[i,j]]['median_income']
        fig = my_hist(df = to_plot, bins = 50, colors= my_hex(),
                        fontsize = 10, fontcolor= my_hex(),
                        xlabel='Median income',
                        ylabel='Frequency',
                        title=list_val[i,j])
st.pyplot(fig)