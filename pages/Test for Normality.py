import streamlit as st 
# https://www.statology.org/normality-test-python/


st.header("Test for Normality in Python (4 Methods)")

code = """
import math
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt

#make this example reproducible
np.random.seed(1)

#generate dataset that contains 1000 log-normal distributed values
lognorm_dataset = lognorm.rvs(s=.5, scale=math.exp(1), size=1000)
"""
st.code(code, language='python')
import math
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt

#make this example reproducible
np.random.seed(1)

#generate dataset that contains 1000 log-normal distributed values
lognorm_dataset = lognorm.rvs(s=.5, scale=math.exp(1), size=1000)

st.markdown(r"""
Many statistical tests make the assumption that datasets are normally distributed.

There are four common ways to check this assumption in Python:

#### 1 : (Visual Method) Create a histogram
If the histogram is roughly “bell-shaped”, then the data is assumed to be normally distributed.
""")
code = """
#create histogram to visualize values in dataset
plt.hist(lognorm_dataset, edgecolor='black', bins=20)
"""
st.code(code, language='python')
fig = plt.figure(figsize=(10,7))
plt.hist(lognorm_dataset, edgecolor='black', bins=20)
st.pyplot(fig)



st.markdown(r"""
By simply looking at this histogram, we can tell the dataset does not exhibit a “bell-shape” and is not normally distributed.

#### 2 : (Visual Method) Create a Q-Q plot.
If the points in the plot roughly fall along a straight diagonal line, then the data is assumed to be normally distributed.
""")

code = """
import statsmodels.api as sm
#create Q-Q plot with 45-degree line added to plot
fig = sm.qqplot(lognorm_dataset, line='45')
"""
st.code(code, language='python')
import statsmodels.api as sm
fig = sm.qqplot(lognorm_dataset, line='45')
st.pyplot(fig)

st.markdown(r"""
If the points on the plot fall roughly along a straight diagonal line, then we typically assume a dataset is normally distributed.

However, the points on this plot clearly don't fall along the red line, so we would not assume that this dataset is normally distributed.

This should make sense considering we generated the data using a log-normal distribution function.
#### 3 : (Formal Statistical Test) Perform a Shapiro-Wilk Test.
If the p-value of the test is greater than $\alpha = 0.05$, then the data is assumed to be normally distributed.
""")

code = """
from scipy.stats import shapiro 
#perform Shapiro-Wilk test for normality
shapiro(lognorm_dataset)
"""
st.code(code, language='python')
from scipy.stats import shapiro 
st.write(shapiro(lognorm_dataset))
st.markdown(r"""
From the output we can see that the test statistic is 0.857 and the corresponding p-value is 3.88e-29 (extremely close to zero).

Since the p-value is less than .05, we reject the null hypothesis of the Shapiro-Wilk test.

This means we have sufficient evidence to say that the sample data does not come from a normal distribution.

#### 4 : (Formal Statistical Test) Perform a Kolmogorov-Smirnov Test.
If the p-value of the test is greater than $\alpha = 0.05$, then the data is assumed to be normally distributed.
""")
code = """
from scipy.stats import kstest
#perform Kolmogorov-Smirnov test for normality
kstest(lognorm_dataset, 'norm')
"""
st.code(code, language='python')
from scipy.stats import kstest
st.write(kstest(lognorm_dataset, 'norm'))

st.markdown(r"""
From the output we can see that the test statistic is 0.841 and the corresponding p-value is 0.0.

Since the p-value is less than .05, we reject the null hypothesis of the Kolmogorov-Smirnov test.

This means we have sufficient evidence to say that the sample data does not come from a normal distribution.
""")


st.header("How to Handle Non-Normal Data")
st.markdown(r"""
If a given dataset is not normally distributed, we can often perform one of the following transformations to make it more normally distributed:

#### 1. Log Transformation: Transform the values from $x$ to $log(x)$.
""")
code = """
#create log-transformed data
data_log = np.log(lognorm_dataset)

#define grid of plots
fig, axs = plt.subplots(nrows=1, ncols=2, figsize = (10,6))

#create histograms
axs[0].hist(lognorm_dataset, edgecolor='black')
axs[1].hist(data_log, edgecolor='black')

#add title to each histogram
axs[0].set_title('Original Data')
axs[1].set_title('Log-Transformed Data')
"""
st.code(code, language='python')
#create log-transformed data
data_log = np.log(lognorm_dataset)

#define grid of plots
fig, axs = plt.subplots(nrows=1, ncols=2, figsize = (10,6))

#create histograms
axs[0].hist(lognorm_dataset, edgecolor='black')
axs[1].hist(data_log, edgecolor='black')

#add title to each histogram
axs[0].set_title('Original Data')
axs[1].set_title('Log-Transformed Data')
st.pyplot(fig)


code = """
shapiro(data_log)
"""
st.code(code, language='python')
st.write(shapiro(data_log))

st.success('P-value in Shapiro-Wilk Test is $>$ then 0.05 \n This means we have sufficient evidence to say that the sample data comes from a normal distribution.', icon="✅")

code = """
kstest(data_log, 'norm')
"""
st.code(code, language='python')
st.write(kstest(data_log, 'norm'))
st.error('P-value in kolmogorov-smirnov test is $<$ then 0.05 \n This means we have still havesufficient evidence to say that the sample data does not come from a normal distribution.', icon="🚨")

st.markdown(r"""
#### 2. Square Root Transformation: Transform the values from $x$ to $\sqrt{x}$.
""")
code ="""
#create log-transformed data
data_log = np.sqrt(lognorm_dataset)

#define grid of plots
fig, axs = plt.subplots(nrows=1, ncols=2)

#create histograms
axs[0].hist(lognorm_dataset, edgecolor='black')
axs[1].hist(data_log, edgecolor='black')

#add title to each histogram
axs[0].set_title('Original Data')
axs[1].set_title('Square Root Transformed Data')
"""
st.code(code, language='python')
#create log-transformed data
data_log = np.sqrt(lognorm_dataset)

#define grid of plots
fig, axs = plt.subplots(nrows=1, ncols=2, figsize = (10,6))

#create histograms
axs[0].hist(lognorm_dataset, edgecolor='black')
axs[1].hist(data_log, edgecolor='black')

#add title to each histogram
axs[0].set_title('Original Data')
axs[1].set_title('Square Root Transformed Data')
st.pyplot(fig)


code = """
shapiro(data_log)
"""
st.code(code, language='python')
st.write(shapiro(data_log))

code = """
kstest(data_log, 'norm')
"""
st.code(code, language='python')
st.write(kstest(data_log, 'norm'))

st.error('Notice how the square root transformed data is much more normally distributed than the original data. But P-value in kolmogorov-smirnov test and Shapiro-Wilk Test is still $<$ then 0.05 \n This means we have still havesufficient evidence to say that the sample data does not come from a normal distribution.', icon="🚨")



st.markdown(r"""
#### 3. Cube Root Transformation: Transform the values from $x$ to $x^{1/3}$.

By performing these transformations, the dataset typically becomes more normally distributed.
""")
code = """
#create Cube Root Transformation
data_log = np.cbrt(lognorm_dataset)

#define grid of plots
fig, axs = plt.subplots(nrows=1, ncols=2)

#create histograms
axs[0].hist(lognorm_dataset, edgecolor='black')
axs[1].hist(data_log, edgecolor='black')

#add title to each histogram
axs[0].set_title('Original Data')
axs[1].set_title('Cube Root Transformed Data')
"""
st.code(code, language='python')
#create Cube Root Transformation
data_log = np.cbrt(lognorm_dataset)

#define grid of plots
fig, axs = plt.subplots(nrows=1, ncols=2, figsize = (10,6))

#create histograms
axs[0].hist(lognorm_dataset, edgecolor='black')
axs[1].hist(data_log, edgecolor='black')

#add title to each histogram
axs[0].set_title('Original Data')
axs[1].set_title('Cube Root Transformed Data')
st.pyplot(fig)

code = """
shapiro(data_log)
"""
st.code(code, language='python')
st.write(shapiro(data_log))

code = """
kstest(data_log, 'norm')
"""
st.code(code, language='python')
st.write(kstest(data_log, 'norm'))

st.error('Notice how the square root transformed data is much more normally distributed than the original data. But P-value in kolmogorov-smirnov test and Shapiro-Wilk Test is still $<$ then 0.05 \n This means we have still havesufficient evidence to say that the sample data does not come from a normal distribution.', icon="🚨")



st.success("Fit your model to the data and NOT your data to the model", icon="✅")