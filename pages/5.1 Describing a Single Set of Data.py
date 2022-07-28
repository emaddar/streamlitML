# Contents of ~/my_app/streamlit_app.py
from random import randint
from matplotlib.pyplot import hist, show
import streamlit as st

st.set_page_config(
    page_icon=":blue_heart:",
    layout="wide"
)

st.header("Visulaize one Numerical Variable")

st.markdown(r"""

####  Histograme
Histograms are useful for showing patterns within your data and getting an idea of the distribution of your variable at a glance.

""")
st.write("Click [here](https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0) for more examples")

x = st.text_input('input x :', '5,7,8,7,2,17,2,9,4,11,12,9,6')
x = [float(i) for i in list(x.split(","))]
bin = st.number_input('input binwidth ', min_value=1, max_value=len(x), value=2, step=1)

import matplotlib.pyplot as plt
fig = plt.figure()
plt.hist(x, bins = int(len(x)/bin))
st.pyplot(fig)


code = """
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]

import matplotlib.pyplot as plt
plt.hist(x, bins = 10)
plt.show()
"""
st.code(code, language='python')


st.markdown(r"""

#####  Histograme distribution modality
The first distinguishing feature apparent in a histogram is the number of **modes**, or **peaks**, in the distribution. A peak occurs anywhere that the distribution falls and then rises again, even if it does not rise as high as the previous peak.
* A **unimodal distribution** only has one peak in the distribution,
* a **bimodal distribution** has two peaks, 
* a **multimodal distribution** has three or more peaks and
* A histogram is described as **uniform** if every value in a dataset occurs roughly the same number of times. 
""")


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings("ignore")

f, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))

# Unimodal
sns.distplot(np.random.normal(10, 5, 10000), ax=ax[0], hist=False, color='blue')
ax[0].set_title('Unimodal', fontsize=14)
ax[0].set_yticklabels([])
ax[0].set_xticklabels([])

# Bimodal
sample_bimodal = pd.DataFrame({'feature1' : np.random.normal(10, 5, 10000),
                   'feature2' : np.random.normal(40, 10, 10000),
                   'feature3' : np.random.randint(0, 2, 10000),
                  })

sample_bimodal['combined'] = sample_bimodal.apply(lambda x: x.feature1 if (x.feature3 == 0 ) else x.feature2, axis=1)

sns.distplot(sample_bimodal['combined'].values, ax=ax[1], color='blue', hist=False)

ax2 = ax[1].twinx()  # instantiate a second axes that shares the same x-axis

sns.distplot(sample_bimodal.feature1, ax=ax2, color='blue', kde_kws={'linestyle':'--'}, hist=False)
sns.distplot((sample_bimodal.feature2), ax=ax2, color='blue', kde_kws={'linestyle':'--'}, hist=False)

f.tight_layout()  # otherwise the right y-label is slightly clipped

ax[1].set_title('Bimodal', fontsize=14)
ax[1].set_yticklabels([])
ax[1].set_xticklabels([])
ax2.set_yticklabels([])


# Multimodal
sample_multi = pd.DataFrame({'feature1' : np.random.normal(10, 5, 10000),
                   'feature2' : np.random.normal(40, 10, 10000),
                   'feature3' : np.random.randint(0, 3, 10000),
                               'feature4' : np.random.normal(80, 4, 10000),
                  })

sample_multi['combined'] = sample_multi.apply(lambda x: x.feature1 if (x.feature3 == 0 ) else (x.feature2 if x.feature3 == 1 else x.feature4), axis=1 )

sns.distplot(sample_multi['combined'].values, ax=ax[2], color='blue', hist=False)

ax3 = ax[2].twinx()  # instantiate a second axes that shares the same x-axis

sns.distplot(sample_multi.feature1, ax=ax3, color='blue', kde_kws={'linestyle':'--'}, hist=False)
sns.distplot((sample_multi.feature2), ax=ax3, color='blue', kde_kws={'linestyle':'--'}, hist=False)
sns.distplot((sample_multi.feature4), ax=ax3, color='blue', kde_kws={'linestyle':'--'}, hist=False)

f.tight_layout()  # otherwise the right y-label is slightly clipped

ax[2].set_title('Multimodal', fontsize=14)
ax[2].set_yticklabels([])
ax[2].set_xticklabels([])
ax3.set_yticklabels([])

st.pyplot(f)




st.write("""
References :

* [Handling Multimodal Distributions & FE Techniques](https://www.kaggle.com/code/iamleonie/handling-multimodal-distributions-fe-techniques/notebook)
* [Histograms](https://sites.utexas.edu/sos/guided/descriptive/numericaldd/descriptiven2/histogram/#:~:text=A%20unimodal%20distribution%20only%20has,data%20is%20skewed%20or%20symmetric.)
* [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/index.html)
""")




st.markdown(r"""

#####  Histograme Skewness
Skewness is the measure of the asymmetry of a histogram (frequency distribution ). A histogram with normal distribution is symmetrical. In other words, the same amount of data falls on both sides of the mean. A normal distribution  will have a skewness of 0. 


""")
st.image(
            "https://upload.wikimedia.org/wikipedia/commons/c/cc/Relationship_between_mean_and_median_under_different_skewness.png",
            width=1000, # Manually Adjust the width of the image as per requirement
        )

st.write("""
References :

* [Handling Multimodal Distributions & FE wikipedia](https://upload.wikimedia.org/wikipedia/commons/c/cc/Relationship_between_mean_and_median_under_different_skewness.png)
""")
