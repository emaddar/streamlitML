import streamlit as st
   
st.set_page_config(
    page_icon=":blue_heart:",
    layout="wide"
)


st.header("**Simple Linear Regression Model**")
st.write(r"""
For each point $(x,y)$ in data set the $y$-value is an independent observation of
$$
y=\beta_1x+\beta_0+\epsilon
$$
where $\beta_1$ and $\beta_0$ are fixed parameters and $\epsilon \sim N(0,\sigma)$.

The line with equation $y=\beta_1x+\beta_0$ is called the **population regression line**.

It is conceptually important to view the model as a sum of two parts: **Deterministic Part** (β1x+β0) describes the trend in $y$ as $x$ increases and **Random Part** ($\epsilon$)  often called the error term or the **_noise_**. This part explains why the actual observed values of $y$ are not exactly on but fluctuate near a line.


There are three parameters in this model: $\beta_0$, $\beta_1$, and $\sigma$. 
Each has an important interpretation, particularly $\beta_1$ and $\sigma$.
The slope parameter $\beta_1$ represents the expected change in $y$ brought about by a unit
 increase in $x$. The standard deviation $\sigma$ represents the magnitude of the noise in the data.
""")




##############################################
##############################################
##############################################  Goodness of Fit of a Straight Line to Data
##############################################
##############################################
st.markdown("")
latex = r'''
#### Goodness of Fit of a Straight Line to Data
Once the scatter diagram of the data has been drawn
and the model assumptions described [here](https://saylordotorg.github.io/text_introductory-statistics/s14-03-modelling-linear-relationships.html)
 at least visually verified (and perhaps the correlation coefficient r computed to quantitatively verify
  the linear trend), the next step in the analysis is to find
   the straight line that best fits the data. 
'''
st.write(latex)

code = """
import matplotlib.pyplot as plt
from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)
print(f"y^ = {round(intercept,2)} + {round(slope,2)}x")     # y^ = 103.11 + -1.75x
"""
st.code(code, language='python')

st.write(r"""
the equation of this line is
$$
\hat{y} = 103.11-1.75x
$$
"""
)

code = """
def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()
"""

st.code(code, language='python')


import matplotlib.pyplot as plt
from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

fig = plt.figure()
plt.rcParams["figure.figsize"] = (20,3)
plt.scatter(x, y)
plt.plot(x, mymodel)
st.pyplot(fig)


st.write(r"""
To each point in the data set there is associated an **error**, 
the positive or negative vertical distance from the point to the line:
positive if the point is above the line and negative if it is below the line.
$$
e = y-\hat{y}
$$
"""
)

code = """
def error(y,y_es):
    return [round(abs(y_i-y_estimated),2) for y_i, y_estimated in zip(y,y_es)]

print(error(y,mymodel))     #[4.65, 4.85, 2.1, 2.85, 11.4, 12.67, 3.4, 0.34, 2.1, 5.84, 5.09, 2.34, 6.6]
"""
st.code(code, language='python')

code = """
import pandas as pd
df = pd.DataFrame(list(zip(x,y,mymodel,error(y,mymodel))), columns=("x", "y", "y^", "e=y-y^"))
print(df)
"""
st.code(code, language='python')

def error(y,y_es):
    return [round(abs(y_i-y_estimated),2) for y_i, y_estimated in zip(y,y_es)]


import pandas as pd
df = pd.DataFrame(list(zip(x,y,mymodel,error(y,mymodel))), columns=("x", "y", "y^", "e=y-y^"))
st.dataframe(df)



##############################################
##############################################
##############################################  Goodness of Fit of a Straight Line to Data
##############################################
##############################################
st.markdown("")
latex = r'''
#### Goodness of Fit of a Straight Line to Data
##### Definition
Given a collection of pairs $(x,y)$ of numbers (in which not all the x-values are the same),
 there is a line $\hat{y} = \hat{\beta_0}+\hat{\beta_1}x$ that best fits the data in the sense
  of minimizing the sum of the squared errors $\sum(y-\hat{y})^2$. 
  It is called the **least squares regression line**. 
  Its slope $\hat{\beta_1}$ and $y$-intercept $\hat{\beta_0}$ are computed using the formulas :
  $$
  \hat{\beta_1} = \frac{SS_{xy}}{SS_{xx}} \textup{    and     } \hat{\beta_0} = \bar{y} - \hat{\beta_1}\bar{x}
  $$
  where
  $$
        SS_{xx}=\sum{x^2}-\frac{1}{n}(\sum x)^2, SS_{xy}=\sum xy-\frac{1}{n}(\sum x)(\sum y)
  $$
The equation $\hat{y} = \hat{\beta_0}+\hat{\beta_1}x$ specifying the least squares regression
 line is called the **least squares regression equation**.
'''
st.write(latex)

code = """

from typing import List
import numpy as np
import math
import scipy.stats

def SumProd(x:List, y:list)->float:
    return sum([x_i*y_i for x_i, y_i in zip(x, y)])

def LSregression(x:List, y:List, alpha:float = 0.05, xp:float =0)-> float:
    assert len(x)==len(y), "x and y must have the same length"
    n= len(x)
    Xmean = np.average(x)
    Ymean = np.average(y)
    Xsumsquar = SumProd(x, x)
    Ysumsquar = SumProd(y, y )

    SSXX = Xsumsquar - 1/n*sum(x)**2
    SSXY = SumProd(x,y)- 1/n*sum(x)*sum(y)
    SSYY = Ysumsquar - 1/n*sum(y)**2
    beta0 = Ymean-SSXY/SSXX*Xmean
    beta1 = SSXY/SSXX
    SSE = SSYY - beta1*SSXY
    Se = math.sqrt(SSE/(n-2))
    r = SSXY/math.sqrt(SSXX*SSYY)
    r2 = beta1*SSXY/SSYY
    t_alpha = scipy.stats.t.ppf(q=1-alpha/2,df=n-2) #find T critical value
    UCI_beta_1 = beta1+t_alpha*(Se/math.sqrt(SSXX))      # for upper Confidence Interval for the Slope β1
    LCI_beta_1 = beta1-t_alpha*(Se/math.sqrt(SSXX))      # for lower Confidence Interval for the Slope β1
    y_est = beta0+(beta1*xp)
    LCIMVy =  y_est - (t_alpha * Se * math.sqrt( (1/n) + ((xp - Xmean)**2/SSXX)))#  For Lower Confidence Interval for the Mean Value of y
    UCIMVy =  y_est + (t_alpha * Se * math.sqrt(1/n + (xp - Xmean)**2/SSXX))#  For Upper Confidence Interval for the Mean Value of y
    LPIINVy = y_est - (t_alpha * Se * math.sqrt(1+ 1/n + ((xp - Xmean)**2/SSXX)))# for Lower Prediction Interval for an Individual New Value of y
    UPIINVy = y_est + (t_alpha * Se * math.sqrt(1+ 1/n + ((xp - Xmean)**2/SSXX)))# for Upper Prediction Interval for an Individual New Value of y

    return [beta0, beta1, SSE, SSXX, Se,  LCI_beta_1, UCI_beta_1, r2, y_est, t_alpha, LCIMVy, UCIMVy, LPIINVy, UPIINVy, y_est, r]

Model = LSregression(x,y)

print(f"Slope = {Model[1]} and Intercept = {Model[0]}" )        # Slope = -1.75 and Intercept = 103.11
"""
st.code(code, language='python')




##############################################
##############################################
##############################################  The Sum of the Squared Errors SSE
##############################################
##############################################
st.markdown("")
latex = r'''
#### The Sum of the Squared Errors $SSE$
In the case of the least squares regression line, 
the line that best fits the data, the sum of the squared errors can
 be computed directly from the data using the following formula :
 $$
SSE = SS_{yy} - \hat{\beta_1} SS_{xy}
 $$
where $SS_{yy} = \sum{y^2}-\frac{1}{n}(\sum y)^2$

**Note** that $\sum(y-\hat{y})^2$ will give you the same results.
'''
st.write(latex)

st.code(r"""print(f"SSE = {round(Model[2],2)}")      # SSE = SSE = 473.07""", language='python')
st.write("Click [here](https://saylordotorg.github.io/text_introductory-statistics/s14-04-the-least-squares-regression-l.html) for more info and applications.")


##############################################
##############################################
##############################################  Sample standard deviation of errors Se
##############################################
##############################################
st.markdown("")
latex = r'''
#### Sample standard deviation of errors $Se$
The statistic $Se$ 
$$
Se = \sqrt{\frac{SSE}{n-2}}
$$
is called the sample standard deviation of errors. It estimates the standard deviation $\sigma$
 of the errors in the population of $y$-values for each fixed value of $x$
'''
st.write(latex)

code = """
print(f"Se = {round(Model[4],2)}")      # Se = 6.56
"""
st.code(code, language='python')




st.header("**Statistical Inferences About the Slope**")
st.write(r"""
The parameter $\beta_1$, the slope of the population regression line, is of primary importance in regression analysis.
For every unit increase in $x$ the mean of the response variable $y$ changes by $\beta_1$ units, increasing if $\beta_1>0$ 
and decreasing if $\beta_1<0$. We wish to construct confidence intervals for $\beta_1$ and test hypotheses about it.
""")


##############################################
##############################################
##############################################  Confidence Intervals for $\beta_1$
##############################################
##############################################
st.markdown("")
latex = r'''
#### Confidence Intervals for $\beta_1$
The slope $\hat{\beta_1}$ of the least squares regression line is a point estimate of $\beta_1$.

A **$100(1−\alpha)\%$ Confidence Interval for the Slope $\beta_1$ of the Population Regression Line** is given by the following formula :
$$
\hat{\beta_1} \pm  t_{\frac{\alpha}{2}}\frac{Se}{\sqrt{SS_{xx}}}
$$
with a number of degrees of freedom $df=n-2$.
'''
st.write(latex)


st.code(r"""print(f"Confidence Intervals for beta1 = [{round(Model[5],2)} , {round(Model[6],2)}]")      # Confidence Intervals for beta1 = [-2.75 , -0.75]""", language='python')






##############################################
##############################################
##############################################  The Coefficient of Determination r²
##############################################
##############################################
st.markdown("")
latex = r'''
#### The Coefficient of Determination $r^2$
The coefficient of determination of a collection of $(x,y)$ pairs is the number $r^2$ computed by any of the following three expressions:
$$
r^2 = \frac{SS_{yy}-SSE}{SS_{yy}} = \frac{SS^2_{xy}}{SS_{xx}SS_{yy}} = \beta_1 \frac{SS_{xy}}{SS_{yy}}
$$
'''
st.write(latex)

st.code(r"""print(f"r² = {round(Model[7],2)}")      # r² = 0.58""",language='python')
st.write(r"""So, about 58% of the variability in the value of $y$ can be explained by $x$.""")







st.header("**Estimation and Prediction**")
st.write(r"""
Read [here](https://saylordotorg.github.io/text_introductory-statistics/s14-07-estimation-and-prediction.html) for more information

**Prediction** is done based on mathematical models ($\hat{y} = \beta_0 + \beta_1x$ in our case)which are defined with the help of parameters which are **estimated** from available data $(x,y)$.

We can use the fitted regression equation to predict the values of new observations.
to predict the value of $y$ given $x=20$, we just need to replace the variable  $x$  with the number 20 in the equation and perform the arithmetic:
$$
\hat{y} = 103.11 - 1.75(20)
$$
""")
st.code(r"""
Model = LSregression(x,y, xp = 20)
print(f"y^ = {round(Model[14],2)}")       # y^ = 68.08
""", language='python')




##############################################
##############################################
##############################################  100(1−α)%  Confidence Interval for the Mean Value of y at x=xp
##############################################
##############################################
st.markdown("")
latex = r'''
#### $100(1−\alpha)\%$  Confidence Interval for the Mean Value of $y$ at $x=xp$

$$
\hat{y} \pm t_{\frac{\alpha}{2}} S_e \sqrt{\frac{1}{n} + \frac{(x_p - \bar{x})^2}{SS_{xx}}}
$$
where

$x_p$ is a particular value of $x$

$\hat{y}_p$  is the numerical value obtained when the least square regression equation is evaluated at $x=x_p$; and

the number of degrees of freedom for $t_{\frac{\alpha}{2}}$ is $df=n-2$.
'''
st.write(latex)
code = r'''
Model = LSregression(x,y, xp=20, alpha=0.05)
print(f"Confidence Interval for the Mean Value of y = [{round(Model[10],2)} , {round(Model[11],2)}]")      # Confidence Interval for the Mean Value of y = [55.09 , 81.07]
'''
st.code(code, language='python')





##############################################
##############################################
##############################################  100(1−α)%  Prediction Interval for an Individual New Value of y at x=xp
##############################################
##############################################
st.markdown("")
latex = r'''
#### $100(1−\alpha)\%$  Prediction Interval for an Individual New Value of $y$ at $x=xp$
The formula for the prediction interval is identical except for the presence of the number 1 underneath the square root sign.
 This means that the prediction interval is always wider than the confidence interval at the same confidence level and value of $x$.
  In practice the presence of the number 1 tends to make it much wider.
$$
\hat{y} \pm t_{\frac{\alpha}{2}} S_e \sqrt{1+ \frac{1}{n} + \frac{(x_p - \bar{x})^2}{SS_{xx}}}
$$
'''

st.write(latex)
code = r'''
Model = LSregression(x,y, xp=20, alpha=0.05)
print(f"Prediction Interval for an Individual New Value of y = [{round(Model[-2],2)} , {round(Model[-1],2)}]")      # Prediction Interval for an Individual New Value of y = [48.66 , 87.5]
'''
st.code(code, language='python')




st.header("**A Complete Example**")
st.write(r"""
Reference [here](https://saylordotorg.github.io/text_introductory-statistics/s14-08-a-complete-example.html)

In general educators are convinced that, all other factors being equal, class attendance has a significant bearing on course performance. To investigate the relationship between attendance and performance, an education researcher selects for study a multiple section introductory statistics course at a large university. Instructors in the course agree to keep an accurate record of attendance throughout one semester. At the end of the semester 26 students are selected a random. For each student in the sample two measurements are taken: x, the number of days the student was absent, and y, the student’s score on the common final exam in the course.


""")

Absences= [2,7,2,7,2,7,0,0,6,6,2,2,1,4,5,4,0,1,0,1,3,1,3,1,3,1]
Score= [76,29,96,63,79,71,88,92,55,70,80,75,63,41,63,88,98,99,89,96,90,90,68,84,80,78]

df = pd.DataFrame(list(zip(Absences, Score)), columns=("Absences","Score"))
st.dataframe(df)


code = """
Model = [round(x,2) for x in LSregression(Absences,Score, xp = 5, alpha=0.05)]
print(f"Beta0 = {Model[0]}")        # Beta0 = 91.24
print(f"Beta1 = {Model[1]}")        # Beta1 = -5.23
print(f"SSE = {Model[2]}")          # SSE = 3819.18
print(f"SSXX = {Model[3]}")         # SSXX = 135.12
print(f"Se = {Model[4]}")           # Se = 12.61
print(f"Confidence Intervals for beta1 = [{Model[5]} , {Model[6]}]")      # Confidence Intervals for beta1 = [-7.47 , -2.99]
print(f"r² = {Model[7]}")           # r² = 0.49
print(f"y^ = {Model[14]}")          # y^ = 65.1
print(f"Confidence Interval for the Mean Value of y = [{Model[10]} , {Model[11]}]") # Confidence Interval for the Mean Value of y = [57.9 , 72.3]
print(f"Prediction Interval for an Individual New Value of y = [{round(Model[-2],2)} , {round(Model[-1],2)}]")      # Prediction Interval for an Individual New Value of y = [65.1 , -0.7]
"""

st.code(code, language='python')

st.markdown("---")
st.header("TRY IT WITH YOUR DATA")
st.markdown("---")



from typing import List
import numpy as np
import math
import scipy.stats

def SumProd(x:List, y:list)->float:
    return sum([x_i*y_i for x_i, y_i in zip(x, y)])

def LSregression(x:List, y:List, alpha:float = 0.05, xp:float =0)-> float:
    assert len(x)==len(y), "x and y must have the same length"
    n= len(x)
    Xmean = np.average(x)
    Ymean = np.average(y)
    Xsumsquar = SumProd(x, x)
    Ysumsquar = SumProd(y, y )

    SSXX = Xsumsquar - 1/n*sum(x)**2
    SSXY = SumProd(x,y)- 1/n*sum(x)*sum(y)
    SSYY = Ysumsquar - 1/n*sum(y)**2
    beta0 = Ymean-SSXY/SSXX*Xmean
    beta1 = SSXY/SSXX
    SSE = SSYY - beta1*SSXY
    Se = math.sqrt(SSE/(n-2))
    r = SSXY/math.sqrt(SSXX*SSYY)
    r2 = beta1*SSXY/SSYY
    t_alpha = scipy.stats.t.ppf(q=1-alpha/2,df=n-2) #find T critical value
    UCI_beta_1 = beta1+t_alpha*(Se/math.sqrt(SSXX))      # for upper Confidence Interval for the Slope β1
    LCI_beta_1 = beta1-t_alpha*(Se/math.sqrt(SSXX))      # for lower Confidence Interval for the Slope β1
    y_est = beta0+(beta1*xp)
    LCIMVy =  y_est - (t_alpha * Se * math.sqrt( (1/n) + ((xp - Xmean)**2/SSXX)))#  For Lower Confidence Interval for the Mean Value of y
    UCIMVy =  y_est + (t_alpha * Se * math.sqrt(1/n + (xp - Xmean)**2/SSXX))#  For Upper Confidence Interval for the Mean Value of y
    LPIINVy = y_est - (t_alpha * Se * math.sqrt(1+ 1/n + ((xp - Xmean)**2/SSXX)))# for Lower Prediction Interval for an Individual New Value of y
    UPIINVy = y_est + (t_alpha * Se * math.sqrt(1+ 1/n + ((xp - Xmean)**2/SSXX)))# for Upper Prediction Interval for an Individual New Value of y

    return [beta0, beta1, SSE, SSXX, Se,  LCI_beta_1, UCI_beta_1, r2, y_est, t_alpha, LCIMVy, UCIMVy, LPIINVy, UPIINVy, y_est, r]
    




col1, col2, col3 = st.columns(3)
with col1:
    x = st.text_input('input x :', '5,7,8,7,2,17,2,9,4,11,12,9,6')
    x = [float(i) for i in list(x.split(","))]
    y = st.text_input('input y :', '99,86,87,88,111,86,103,87,94,78,77,85,86')
    y = [float(i) for i in list(y.split(","))]
    xp = st.text_input('input xp :', '20')
    xp = float(xp)
    alpha = st.text_input('input aplha :', '0.05')
    alpha = float(alpha)
 

click = col1.button("Calculate") 
import time
if click:
    Model = [round(x,2) for x in LSregression(Absences,Score, xp, alpha)]
    #Model = LSregression(Absences,Score, xp, alpha)
    col2.write(f"Beta0 = {Model[0]}")        # Beta0 = 91.24
    time.sleep(0.11)
    col2.write(f"Beta1 = {Model[1]}")        # Beta1 = -5.23
    time.sleep(0.11)
    col2.write(f"SSE = {Model[2]}")          # SSE = 3819.18
    time.sleep(0.11)
    col2.write(f"SSXX = {Model[3]}")         # SSXX = 135.12
    time.sleep(0.11)
    col2.write(f"Se = {Model[4]}")           # Se = 12.61
    time.sleep(0.11)
    col2.write(f"Confidence Intervals for beta1 = [{Model[5]} , {Model[6]}]")      # Confidence Intervals for beta1 = [-7.47 , -2.99]
    time.sleep(0.11)
    col2.write(f"r² = {Model[7]}")           # r² = 0.49
    time.sleep(0.11)
    col2.write(f"y^ = {Model[14]}")          # y^ = 65.1
    time.sleep(0.11)
    col2.write(f"Confidence Interval for the Mean Value of y = [{Model[10]} , {Model[11]}]") # Confidence Interval for the Mean Value of y = [57.9 , 72.3]
    time.sleep(0.11)
    col2.write(f"Prediction Interval for an Individual New Value of y = [{round(Model[-2],2)} , {round(Model[-1],2)}]")      # Prediction Interval for an Individual New Value of y = [65.1 , -0.7]

    with col3:
        def myfunc(x):
            return Model[1] * x + Model[0]

        mymodel = list(map(myfunc, x))
        import matplotlib
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        fig1 = plt.figure()
        plt.rcParams["figure.figsize"] = (10,5)
        plt.scatter(x, y)
        plt.plot(x, mymodel)
        st.pyplot(fig1)

    