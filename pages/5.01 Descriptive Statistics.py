# Contents of ~/my_app/streamlit_app.py
from nbformat import write
import streamlit as st

st.set_page_config(
    page_icon=":blue_heart:",
    layout="wide"
)

st.header("**Some statistics**")
st.write("You start generating some statistics. Probably the simplest statistic is the number of data points:")

code = """
My_list = [9, 0, 6, 7, 9, 0, 3, 7, 7, 4, 10, 2, 0, 8, 7, 5, 1, 3, 5, 0]
num_points = len(My_list)           #20
"""
st.code(code, language='python')

st.write("You’re probably also interested in the largest and smallest values:")

code = """
largest_value = max(My_list)        #10
smallest_value = min(My_list)       #0
"""
st.code(code, language='python')


st.write("which are just special cases of wanting to know the values in specific positions:")

code = """
sorted_values = sorted(My_list)
smallest_value = sorted_values[0] # 0
second_smallest_value = sorted_values[1] # 0
second_largest_value = sorted_values[-2] # 9
"""
st.code(code, language='python')

############################################################################################
############################################################################################
#                                    Central tendancey                                      #
############################################################################################
############################################################################################
st.header('Central Tendencies')


##############################################
##############################################
##############################################   Average
##############################################
##############################################

st.write(r"""
#### Average
""")
st.write("Most commonly we’ll use the **mean** (or **average**), which is just the sum of the data divided by its count:")
st.latex(r"\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i  =  \frac{1}{n} (x_1 + \cdots + x_n)")

st.write('You can create your own function :')
code = """
from typing import List
def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)

mean(My_list)   #   4.65
"""
st.code(code, language='python')

st.write("For Python 3+, use below. (Fastest.) ")
code = """
sum(My_list) / len(My_list)     #4.65
"""
st.code(code, language='python')

st.write("For Python 3.4+, use  **statistics.mean** for numerical stability with floats. (Slow.):")
code = """
import statistics
statistics.mean(My_list)     ##4.65
"""
st.code(code, language='python')


st.write('For Python 3.8+, use **statistics.fmean** for numerical stability with floats. (Fast.)')
code = """
statistics.mean(My_list)     #4.65
"""
st.code(code, language='python')

st.write("we can also use **numpy.mean** :")
code = """
import numpy
numpy.mean(My_list)     #4.65
"""
st.code(code, language='python')


##############################################
##############################################
##############################################   Median
##############################################
##############################################

st.markdown(' ')
st.write(r"""
#### Median
""")
st.markdown("We’ll also sometimes be interested in the **median** :")

# https://discuss.streamlit.io/t/markdown-and-math-equations/9990
latext = r'''
if $n$ is odd : 
$$
{\displaystyle \mathrm {median} (x)=x_{(n+1)/2}}
$$
if $n$ is even : 
$$
{\displaystyle \mathrm {median} (x)={ \frac{x_{(n/2)}+x_{(n/2)+1}}{2}}}
$$
'''
st.write(latext)


st.write('We’ll write different functions for the even and odd cases and combine them:')

code = '''
from typing import List

def median_odd(xs: List[float]) -> float:
    """If len(xs) is odd, the median is the middle element"""
    return sorted(xs)[len(xs) // 2]
def median_even(xs: List[float]) -> float:
    """If len(xs) is even, it's the average of the middle two elements"""
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) // 2 # e.g. length 4 => hi_midpoint 2
    return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint]) / 2
def median(v: List[float]) -> float:
    """Finds the 'middle-most' value of v"""
    return median_even(v) if len(v) % 2 == 0 else median_odd(v)

median(My_list)     #5.0
'''
st.code(code, language='python')

st.markdown('Python 3.4 has **statistics.median**:')
code = """
import statistics
statistics.median(My_list)     #5.0
"""
st.code(code, language='python')

st.markdown('Or **numpy.median**:')
code = """
import numpy
numpy.median(My_list)     #5.0
"""
st.code(code, language='python')


##############################################
##############################################
##############################################   Mode
##############################################
##############################################
st.markdown("")
st.write(r'''
#### Mode
The **mode** is the value that appears most often in a set of data values.
''')

code = r"""
from collections import Counter
from typing import List
def My_mode(x: List) -> float : # The Counter class in the collections package is used to count the number of occurrences of each element present in the given data set.
     return Counter(My_list).most_common(1)[0][0] #The .most_common() method of the Counter class returns a list containing two-items tuples with each unique element and its frequency.

My_mode(My_list)        #0
"""
st.code(code, language='python')


st.write('Or :')
code = """
def mode(x: List[float]) -> List[float]:
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()
    if count == max_count]

mode(My_list)
"""
st.code(code, language='python')

st.write('And as usual, we can use **statistics.mode**:')
code = """
import statistics
statistics.mode(My_list)        #0
"""

st.code(code, language='python')

st.write(r"""
##### Important :warning: :
""")
st.write("**numpy** by itself does not support any such functionality. Instead, can you use **scipy.stats.mode**  (Slow.)")
code = '''
import scipy.stats
print(scipy.stats.mode(My_list)[0])
'''
st.code(code, language='python')



##############################################
##############################################
##############################################   Quantile,
##############################################
##############################################
st.markdown("")
st.write(r'''
#### Quantile,
The **quantile,** generalization of the median, it represents the value under which a certain percentile of the data lies (the median represents the value under which 50% of the data lies):
''')

code = """
def quantile(xs: List[float], p: float) -> float:
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]

print("q_10 = ", quantile(My_list, 0.10))       #0
print("Q1 =q_25 = ", quantile(My_list, 0.25))   #2
print("Q3 = q_75 = ", quantile(My_list, 0.75))  #7
print("q_90 = ", quantile(My_list, 0.90))       #9
"""
st.code(code, language='python')

code = """
import numpy
numpy.quantile(My_list, 0.25)
numpy.quantile(My_list, 0.50)
numpy.quantile(My_list, 0.75)
"""
st.code(code, language='python')


code = """
import statistics
print(statistics.quantiles(My_list, method='inclusive'))
"""
st.code(code, language='python')

st.write(f"""
##### Important :warning:
The built-in statistics.quantiles' default method is “exclusive”, however the numpy.quantile is inclusive. check out this [docs](https://docs.python.org/3/library/statistics.html#statistics.quantiles)  to find out which one suits your needs.
""")
st.write("")

code = """
from scipy.stats.mstats import mquantiles
print(mquantiles(My_list))
"""
st.code(code, language='python')

st.markdown("")
st.markdown("")

st.write(r"**Centiles**/**percentiles** are descriptions of quantiles relative to 100; so the 75th percentile (upper quartile) is 75% or three quarters of the way up an ascending list of sorted values of a sample. The 25th percentile (lower quartile) is one quarter of the way up this rank order")
code = """
import numpy
print(numpy.percentile(My_list, 25))
print(numpy.percentile(My_list, 50))
print(numpy.percentile(My_list, 75))
"""
st.code(code, language='python')




############################################################################################
############################################################################################
#                                    Dispersion tendancey                                      #
############################################################################################
############################################################################################
st.header('Dispersion Tendencies')
st.write("**Dispersion** is the degree to which data is distributed **around this central tendency**, and is represented by range, deviation, variance, standard deviation and standard error...")

##############################################
##############################################
##############################################   Range,
##############################################
##############################################
st.markdown("")
latex = r'''
#### Range
A very simple measure is the **range**, 
which is just the difference between the largest and smallest elements:
$$
R = Max_x - Min_x
$$
'''
st.write(latex)

code = """
from typing import List
# "range" already means something in Python, so we'll use a different name
def data_range(xs: List[float]) -> float:
    return max(xs) - min(xs)

data_range(My_list)     #10
"""
st.code(code, language='python')
st.write('Or simply :')
st.code("max(My_list) - min(My_list)     #10", language='python')

##############################################
##############################################
##############################################   Variance,
##############################################
##############################################
st.markdown("")
latex = r'''
#### Variance
A more complex measure of dispersion is the variance, the population variance is computed as:
$$
{\displaystyle \sigma^2(X)={\frac {1}{n}}\sum _{i=1}^{n}(x_{i}-\mu )^{2}}
$$
and the sample variance is computes as :
$$
{\displaystyle S^2(X)={\frac {1}{n-1}}\sum _{i=1}^{n}(x_{i}-\mu )^{2}}
$$
where ${\displaystyle \mu }$  is the average value.

'''
st.write(latex)

code = '''
def va(x:List) -> float:
    ave = mean(x)
    de = [x_i-ave for x_i in x]
    return sum([x_i**2 for x_i in de])

def P_variance(x:List) -> float:
    de = va(x)
    return de/(len(My_list))

def S_variance(x:List)-> float:
    de = va(x)
    return de/(len(My_list)-1)

print("S_variance =",S_variance(My_list))       #11.292105263157893
print("P_variance = ",P_variance(My_list))      #10.7275
'''
st.code(code, language='python')

st.write('Using **numpy** or **statistics** packages, we can find the same results :')

code = """
print("P_Variance with Statistics =",statistics.pvariance(My_list))
print("S_Variance with Statistics =",statistics.variance(My_list))

print("P_variance with Numpy = ",numpy.var(My_list))
print("S_variance with Numpy = ",numpy.var(My_list, ddof = 1))
"""
st.code(code, language='python')



##############################################
##############################################
##############################################   Standard deviation,
##############################################
##############################################
st.markdown("")
latex = r'''
#### Standard deviation
The standard deviation is the square root of its variance.
$$
{\displaystyle \sigma =  \sqrt{\sigma^2(X)}}
'''
st.write(latex)

code = """
import math
def Pstde(xs: List[float]) -> float:
    return math.sqrt(P_variance(xs))

def Sstde(xs: List[float]) -> float:
    return math.sqrt(S_variance(xs))

print("Population standard deviation = ",Pstde(My_list))      #3.2752862470324633
print("Sample standard deviation =",Sstde(My_list))       #3.3603727863375354
"""
st.code(code, language='python')


st.write('Using **numpy** or **statistics** packages, we can also find the same results :')
code = """
print("P_stdev with Statistics =",statistics.pstdev(My_list))
print("S_stdev with Statistics =",statistics.stdev(My_list))

print("P_stdev with Numpy = ",numpy.std(My_list))
print("S_stdev with Numpy = ",numpy.std(My_list, ddof = 1))
"""

st.code(code, language='python')





##############################################
##############################################
##############################################   Interquartile range
##############################################
##############################################
st.markdown("")
latex = r'''
#### Interquartile range
A more robust alternative computes the difference between the 75th percentile value and the 25th percentile value.
The interquartile range tells you the spread of the middle half of your distribution:
$$
IQR =  Q_3 - Q_1
$$
'''
st.write(latex)


code = """
q75, q25 = numpy.percentile(My_list, [75 ,25])
print("IQR =", q75-q25)
"""
st.code(code, language='python')

st.write("Or you can easily use  **iqr** function in **scipy.stats** (Slow.):")
code = """
import scipy.stats
print("IQR = ",scipy.stats.iqr(My_list))
"""
st.code(code, language='python')




##############################################
##############################################
##############################################   Coefficient of variation
##############################################
##############################################
st.markdown("")
latex = r'''
#### Coefficient of variation
The coefficient of variation (CV) is the ratio of the standard deviation to the mean. The higher the coefficient of variation, the greater the level of dispersion around the mean. It is generally expressed as a percentage.:
$$
CV_x =  \frac{\sigma_x}{\bar{x}}
$$
where $\bar{x}$ is the average value and $\sigma$ is the population standard deviation.
'''
st.write(latex)

code = """
def CV(xs: List[float]) -> float:
    return Pstde(xs)/mean(xs)*100

print("CV = ", CV(My_list))     #CV =  70.43626337704222
"""

st.code(code, language='python')
st.write("Or")
st.code("""print("CV =", numpy.std(My_list)/numpy.average(My_list)*100)""", language='python')
st.write("Or")
st.code("""print("CV = ", statistics.pstdev(My_list)/statistics.mean(My_list)*100)""", language='python')



############################################################################################
############################################################################################
#                                    Correlation                                          #
############################################################################################
############################################################################################
st.header('Correlation')

st.markdown("")

##############################################
##############################################
##############################################   Covariance
##############################################
##############################################
st.markdown("")
latex = r'''
#### Covariance
The covariance between variable $x$ and $y$ is the mean value of the product of the deviations of two variates from their respective means :
$$
cov_{x,y}=\frac{\sum_{i=1}^{n}(x_{i}-\bar{x})(y_{i}-\bar{y})}{n-1}
$$
'''
st.write(latex)

code = """
import random
x=[]
y = []
for i in range(20):
    x.append(random.randint(0,10))
    y.append(random.randint(0,10))

print(numpy.cov(x, y, bias=False)[0][1])
"""

st.code(code, language='python')

latex = r'''
The parameter **bias** : Default normalization (False) is by $(n - 1)$, where $n$ is the number of observations given (unbiased estimate). If bias is True, then normalization is by $N$:
$$
cov_{x,y}=\frac{1}{n}\sum_{i=1}^{n}(x_{i}-\bar{x})(y_{i}-\bar{y})
$$
These values can be overridden by using the keyword ddof in numpy versions >= 1.5.'''
st.write(latex)

##############################################
##############################################
##############################################   Covariance Matrix
##############################################
##############################################
st.markdown("")
latex = r'''
#### Covariance Matrix
The Covariance Matrix (also called Variance-Covariance Matrix) is a square matrix given by $C_{ij} = \sigma(x_i, x_j)$ where $C\in \mathbb{R}^{d\times d}$ and $\sigma(x, x) = \frac{1}{n-1}\sum_{i=1}^{n}(x_{i}-\bar{x})(y_{i}-\bar{y})$. The diagonal entries of the covariance matrix are the variances and the other entries are the covariances. For this reason, the covariance matrix is sometimes called the **_variance-covariance matrix_**. Also the covariance matrix is symmetric since $\sigma(x,y) = \sigma(y,x)$
For two-dimensional case (two variables $X$ and $Y$), following from the previous equations the covariance matrix is given by :
$$
C(X,Y)=\begin{bmatrix}
\sigma(X,X) & \sigma(Y,X)\\ 
\sigma(X,Y) & \sigma(Y,Y)
\end{bmatrix} = \begin{bmatrix}
\sigma^2_X & Cov(Y,X) \\ 
Cov(X,Y)  & \sigma^2_Y
\end{bmatrix}
$$
For more information click [here](https://www.cuemath.com/algebra/covariance-matrix/).
'''
st.write(latex)


code = """
import numpy as np

A = [45,37,42,35,39]
B = [38,31,26,28,33]
C = [10,15,17,21,12]

data = np.array([A,B,C])

covMatrix = np.cov(data,bias=True)
print (covMatrix)
"""
st.code(code, language='python')

st.write(r"""Run the code, and you’ll get the following matrix:
$$
\begin{bmatrix}
12.64 &  7.68& -9.6\\ 
 7.68 &  17.36& -13.8\\ 
-9.6  & -13.8 & 14.8
\end{bmatrix}
$$
""")





##############################################
##############################################
##############################################  Correlation,
##############################################
##############################################
st.markdown("")
latex = r'''
#### Correlation (Pearson $\rho$ correlation coefficient)
Correlation is a statistical term describing the degree to which two variables move in coordination with one another.

The **population correlation coefficient** ${\displaystyle \rho _{X,Y}}$ between two random variables ${\displaystyle X}$ and ${\displaystyle Y}$ with standard deviations ${\displaystyle \sigma _{X}}$ and ${\displaystyle \sigma _{Y}}$ is defined as:
$$
\rho _{X,Y}=\operatorname {corr} (X,Y)={\operatorname {cov} (X,Y) \over \sigma _{X}\sigma _{Y}}
$$

The **sample correlation coefficient** is defined as:
$$
{\displaystyle r_{xy} = \quad {\frac {\sum \limits _{i=1}^{n}(x_{i}-{\bar {x}})(y_{i}-{\bar {y}})}{(n-1)\sigma_{x}\sigma_{y}}}={\frac {\sum \limits _{i=1}^{n}(x_{i}-{\bar {x}})(y_{i}-{\bar {y}})}{\sqrt {\sum \limits _{i=1}^{n}(x_{i}-{\bar {x}})^{2}\sum \limits _{i=1}^{n}(y_{i}-{\bar {y}})^{2}}}}}
$$
'''
st.write(latex)

code = """
experience = [1, 3, 4, 5, 5, 6, 7, 10, 11, 12, 15, 20, 25, 28, 30,35]
salary = [20000, 30000, 40000, 45000, 55000, 60000, 80000, 100000, 130000, 150000, 200000, 230000, 250000, 300000, 350000, 400000]

import numpy as np
print(np.corrcoef(experience, salary)[0][1])
"""

st.code(code, language='python')
st.write("""There are other types of correlation : Spearman's and Kendall's coefficient.
The **scipy.stats** module includes the **pearsonr(x, y)** function to calculate Pearson's correlation coefficient between two data samples.
""")
code = """
import scipy.stats as stats
print(stats.pearsonr (experience, salary)[0])
"""
st.code(code, language='python')

st.write("We could calculate **Spearman's** and **Kendall's coefficient** in the same fashion:")

code = """
import scipy.stats as stats
print(stats.spearmanr(experience, salary)[0])
print(stats.kendalltau(experience, salary)[0])
"""
st.code(code, language='python')





##############################################
##############################################
##############################################  Correlation Matrix
##############################################
##############################################
st.markdown("")
latex = r'''
#### Correlation Matrix
A correlation matrix is simply a table which displays the correlation coefficients for different variables. The matrix depicts the correlation between all the possible pairs of values in a table. It is a powerful tool to summarize a large dataset and to identify and visualize patterns in the given data.
The correlation matrix is a $(K \times K)$ square and symmetrical matrix whose $ij$ entry is the correlation between the columns $i$ and $j$ of $X$.
For two variables $X$ and $Y$, following the previous definition the correlation matrix is given by
$$
\begin{bmatrix}
\rho(X,X) & \rho(X,Y)\\ 
\rho(X,Y) & \rho(Y,Y)
\end{bmatrix} = \begin{bmatrix}
1 & \rho(X,Y)\\ 
\rho(X,Y) &1 
\end{bmatrix}
$$
'''
st.write(latex)

code = """
import numpy
print(numpy.corrcoef(experience, salary))
"""
st.code(code, language='python')

st.write(r"""
$$
\begin{bmatrix}
1 & 0.99298458\\ 
0.99298458 &1 
\end{bmatrix}
$$
""")

st.write(r"""
If we wanted to calculate the correlation between two columns, we could use the $$pandas.corr()$$ method, as follows:
""")

code = """
import pandas as pd
df = pd.DataFrame(list(zip(experience, salary)), columns=['experience', 'salary'])
print(df['experience'].corr(df['salary'], method='pearson'))         #0.9929845761480398
print(df['experience'].corr(df['salary'], method='spearman'))        #0.9992644353546791
print(df['experience'].corr(df['salary'], method='kendall'))         #0.9958246164193105
"""
st.code(code, language='python')

st.write("In case we wanted to explore the correlation between all the pairs of variables, we could simply use the **.corr()** method directly to our DataFrame, which results again in a correlation matrix with the coefficient of all the pairs of variables:")
code = """
print(df.corr())
"""
st.code(code, language='python')



##############################################
##############################################
##############################################  Simpson's paradox
##############################################
##############################################
st.markdown("")
latex = r'''
#### Simpson's paradox
Simpson's paradox, which also goes by several other names, is a phenomenon in probability and statistics in which a trend appears in several groups of data but disappears or reverses when the groups are combined.

See more [here](https://campus.datacamp.com/courses/intermediate-regression-with-statsmodels-in-python/interactions-2?ex=12) and [here](https://www.kaggle.com/code/saicataram/simpson-s-paradox-in-python/notebook)
'''
st.write(latex)
youtub = st.button("Youtube video")
if youtub:
    st.video("https://www.youtube.com/watch?v=t-Ci3FosqZs")





##############################################
##############################################
##############################################  Correlation and Causation
##############################################
##############################################
st.markdown("")
latex = r'''
#### Correlation and Causation

###### _"Association does not imply causation"_
If $x$ and $y$ are strongly correlated, that might mean that $x$
causes $y$, that $y$ causes $x$, that each causes the other, that some third factor
causes both, or nothing at all :stuck_out_tongue_winking_eye:.

**_Causation is when one thing deirectly influences another_**
'''
st.write(latex)

youtub1 = st.button("See Youtube video")
if youtub1:
    st.video("https://www.youtube.com/watch?v=7bT17r_yIrw")

st.markdown("---")
st.header("Descriptive Statistics")
st.markdown("---")

col1, col2, col3, col4  = st.columns(4)
with col1:
    len_x = int(st.number_input("Length of X = ", 2,5000,500))
    len_y = int(len_x)
    st.write("Length of Y = ", len_y)
    import random
    import pandas as pd
    x= []
    y= []
    for i in range(len_x):
        x.append(random.randint(0,100))
        y.append(random.randint(0,100))
    df = pd.DataFrame(list(zip(x, y)), columns=['X', 'Y'])
    st.dataframe(df)

with col2:
    st.markdown("Descriptive Statistics")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.write(df.X.describe())

with col3:
    st.markdown("Descriptive Statistics")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.write(df.Y.describe())


with col4:
    st.markdown("**Correlation Matrix**")
    st.write("Pearson correlation = ",df['X'].corr(df['Y'], method='pearson'))
    st.write("Pearson correlation = ",df['X'].corr(df['Y'], method='spearman'))
    st.write("Pearson correlation = ",df['X'].corr(df['Y'], method='kendall'))
    st.markdown("")
    st.markdown("")
    st.markdown("")
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.scatter(df.X, df.Y)
    st.pyplot(fig)


st.write("For more, read [this](https://fxjollois.github.io/cours-2019-2020/m1--add-massives/seance2-stats.html) article (in frensh)")
