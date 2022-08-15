# Contents of ~/my_app/streamlit_app.py
from cProfile import label
import streamlit as st

st.set_page_config(
    page_icon=":blue_heart:",
)


st.markdown("<h1 style='text-align: center; color: black;'>Continuous Distributions</h1>", unsafe_allow_html=True)
st.markdown(r"""
As a reminder, a random variable $X$ has an associated probability distribution $F(·)$, also
know as a cumulative distribution function (**CDF**), which is a function from the sample
space $S$ to the interval $[0, 1]$, i.e., $F : S → [0, 1]$. For any given $x ∈ S$, the CDF returns
the probability $F(x) = P(X ≤ x)$, which uniquely defines the distribution of $X$. In general,
the CDF can take any form as long as it defines a valid probability statement, such that
$0 ≤ F(x) ≤ 1$ for any $x ∈ S$ and $F(a) ≤ F(b)$ for all $a ≤ b$.
As another reminder, a probability distribution has an associated function $f(·)$ that
is referred to as a probability mass function (PMF) or probability distribution function
(**PDF**). 

#### Definition : 
In statistics, a parameter $θ = t(F)$ refers to a some function of a probability
distribution that is used to characterize the distribution. For example, the expected value
 $µ = E(X)$ and the variance $\sigma^2 = E((X - µ)^2)$
are parameters that are commonly used to
describe the location and spread of probability distributions.

The mean indicates where the bell is centered, and the
standard deviation how “wide” it is.

""")

st.markdown("<h2 style='text-align: left; color: black;'>Normal Distribution</h2>", unsafe_allow_html=True)

st.markdown(r"""
The normal (or **Gaussian**) distribution is the most well-known and commonly used probability distribution. 
The normal distribution is quite important because of the **central limit
theorem**, which is discussed in the following section. The normal distribution is a family of
probability distributions defined by two parameters: the mean $\mu$ and the variance $\sigma^2$.


The normal distribution has the properties:
* PDF : ${\displaystyle f(x)={\frac {1}{\sigma {\sqrt {2\pi }}}}e^{-{\frac {1}{2}}\left({\frac {x-\mu }{\sigma }}\right)^{2}}}$
* CDF : $F(x) = \Phi\left ( \frac{x-\mu}{\sigma} \right )$, where $\Phi(x) = \frac{1}{\sqrt{2\pi}}\int_{\infty}^xe^{\frac{-z^2}{2}}dz = {\displaystyle {\frac {1}{2}}\left[1+\operatorname {erf} \left({\frac {x-\mu }{\sigma {\sqrt {2}}}}\right)\right]}$ and ${\displaystyle \operatorname {erf} u={\frac {2}{\sqrt {\pi }}}\int _{0}^{u}e^{-t^{2}}\,dt.}$
* Mean: $E(X) = \mu$
* Variance : $\textup{Var}(x)=\sigma^2$


To denote that $X$ follows a normal distribution with mean $\mu$ and variance $\sigma^2$
, it is typical
to write 
$$
X \sim N(\mu, \sigma^2)
$$ 
where the $\sim$ symbol should be read as "is distributed as". 
""")


code = """
import math
SQRT_TWO_PI = math.sqrt(2 * math.pi)

def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (math.exp(-(x-mu) ** 2 / (2 *sigma ** 2)) / (SQRT_TWO_PI * sigma))    


def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

import matplotlib.pyplot as plt
xs = [x / 10.0 for x in range(-50, 50)]
fig, ax = plt.subplots(1,2,figsize=(10,5))
plt.sca(ax[0])

plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_pdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend()
plt.ylabel('f(x)')
plt.xlabel('x')
plt.title("Various Normal pdfs")


plt.sca(ax[1])
plt.plot(xs,[normal_cdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[normal_cdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[normal_cdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_cdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend(loc=4) # bottom right
plt.ylabel('F(x)')
plt.xlabel('x')
plt.title("Various Normal cdfs")
plt.show()
"""

st.code(code, language='python')


import math
SQRT_TWO_PI = math.sqrt(2 * math.pi)

def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (math.exp(-(x-mu) ** 2 / (2 *sigma ** 2)) / (SQRT_TWO_PI * sigma))    


def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

import matplotlib.pyplot as plt

xs = [x / 10.0 for x in range(-50, 50)]


fig, ax = plt.subplots(1,2,figsize=(10,5))
plt.sca(ax[0])

plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_pdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend()
plt.ylabel('f(x)')
plt.xlabel('x')
plt.title("Various Normal pdfs")


plt.sca(ax[1])
plt.plot(xs,[normal_cdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[normal_cdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[normal_cdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_cdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend(loc=4) # bottom right
plt.ylabel('F(x)')
plt.xlabel('x')
plt.title("Various Normal cdfs")
st.pyplot(fig)


st.markdown(r"""
#### Try it yourself :
""")

xmin = st.number_input('x from :', -1000000, 1000000, value=-5)*10
xmax = st.number_input('x to :', -1000000, 1000000, value=5)*10


xs = [x / 10.0 for x in range(int(xmin), int(xmax))]

mu = st.number_input('mu = ', -1000000, 1000000, value=0)
sigma = st.number_input('sigma = ', 1)



fig, ax = plt.subplots(1,2,figsize=(10,5))
plt.sca(ax[0])

plt.plot(xs,[normal_pdf(x,sigma=sigma, mu = mu) for x in xs],'-',label=f'mu={mu},sigma={sigma}')
plt.legend()
plt.ylabel('f(x)')
plt.xlabel('x')
plt.title("Various Normal pdfs")


plt.sca(ax[1])
plt.plot(xs,[normal_cdf(x,sigma=sigma, mu = mu) for x in xs],'-',label=f'mu={mu},sigma={sigma}')
plt.legend(loc=4) # bottom right
plt.ylabel('F(x)')
plt.xlabel('x')
plt.title("Various Normal cdfs")
st.pyplot(fig)


st.markdown(r"""
We can also use `scipy` library to calculate and plot Normal pdf and cdf Probabilities ([ref](https://www.statology.org/normal-cdf-in-python/))
""")
code = """
from scipy.stats import norm

#calculate probability that random value is less than 1.96 in normal CDF
norm.cdf(1.96)
"""
st.code(code, language='python')


from scipy.stats import norm

#calculate probability that random value is less than 1.96 in normal CDF
st.write(norm.cdf(1.96))

st.markdown(r"""
The probability that a random variables takes on a value less than 1.96 in a standard normal distribution is roughly 0.975.

We can also find the probability that a random variable takes on a value greater than 1.96 by simply subtracting this value from 1:
""")

code = """
#calculate probability that random value is greater than 1.96 in normal CDF
1 - norm.cdf(1.96)
"""
st.code(code, language='python')

st.write(1 - norm.cdf(1.96))

st.markdown("""The probability that a random variables takes on a value greater than 1.96 in a standard normal distribution is roughly 0.025.

The following code shows how to plot a normal CDF and pdf in Python:
""")

code = """
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss


x = np.linspace(-4, 4, 1000)
pdf = ss.norm.pdf(x,loc=0,scale=1)
cdf = ss.norm.cdf(x,loc=0,scale=1)

fig, ax = plt.subplots(1,2,figsize=(10,5))
plt.sca(ax[0])
#plot normal PDF
plt.plot(x, pdf, label= 'mean=0 and sigma=1')
plt.legend()
plt.ylabel('f(x)')
plt.xlabel('x')
plt.title("PDF")

plt.sca(ax[1])
#plot normal CDF
plt.plot(x, cdf, label= 'mean=0 and sigma=1')
plt.legend()
plt.ylabel('F(x)')
plt.xlabel('x')
plt.title("CDF")

plt.show()
"""

st.code(code, language='python')

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss


x = np.linspace(-4, 4, 1000)
pdf = ss.norm.pdf(x,loc=0,scale=1)
cdf = ss.norm.cdf(x,loc=0,scale=1)

fig, ax = plt.subplots(1,2,figsize=(10,5))
plt.sca(ax[0])
#plot normal PDF
plt.plot(x, pdf, label= 'mean=0 and sigma=1')
plt.legend()
plt.ylabel('f(x)')
plt.xlabel('x')
plt.title("PDF")

plt.sca(ax[1])
#plot normal CDF
plt.plot(x, cdf, label= 'mean=0 and sigma=1')
plt.legend()
plt.ylabel('F(x)')
plt.xlabel('x')
plt.title("CDF")

st.pyplot(fig)

