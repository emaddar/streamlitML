# Contents of ~/my_app/streamlit_app.py
import streamlit as st

st.set_page_config(
    page_icon=":blue_heart:",
    layout="wide"
)
#Probability number (between 0 and 1) used to quantify the likelihood, or chance, that an outcome of a random

st.header("Probability")

st.markdown(r"""

**Probability**: number (between 0 and 1) used to quantify the likelihood, or chance, 
that an outcome of a random expirement will occur.

- We can obtain probability from :
    * Subjective experience
    * Observations and Data collection
    * Computations using mathematical rules

#### Random Expirement
- An expirement that can result in **_different outcomes_**, even though it is repeated the same manner every time.
- An outcome is the result of a **_singel trail_**.
- Sample Space $S$ : is the set of **_all possible outcomes_**.
- Event : subject of $S$.

##### Example : Rolling a die once
If you roll a die once, The possible outcomes are 1, 2, 3, 4, 5, or 6.
$$
S = \{1,2,3,4,5,6\}
$$

Event $A$ : getting a number greater than 3
$$
A = \{4,5,6 \}
$$
If $P(A)$ denotes the probability of the event $A$, then
$$
P(A) = \frac{\textup{Number of occurence of A substring}}{\textup{Total number of possible outcomes}} = \frac{n(A)}{n(S)} = \frac{2}{6}
$$
This probability is called : **_Theoretical Probability_**


#### Empirical vs Theoretical Probability

Suppose a coin is flipped 4 times, and we got 1 head and 3 tails. 
So, the probability of getting a head, in this case, is empirical and its value is $\frac{1}{4}$.
 But, we know that when a coin is tossed, the probability to get a head is $\frac{1}{2}$.
  This is theoretical probability.

#####  Note: 
We must remember that empirical probability is also called experimental probability. Also, we must know that the value of empirical probability may or may not be equal to theoretical probability. But, when the experiment is repeated a large number of times, then empirical probability tends to close up on theoretical probability.
""")


for i in range(5):
    st.markdown("")

n = int(st.number_input("No. of Trails = ", 1,1000000,5000))

import random
import matplotlib.pyplot as plt 

results = {
    'heads': 0,
    'tails': 0,
}
sides = list(results.keys())
prob = []
for i in range(n):
    results[random.choice(sides)] += 1
    prob.append(results['heads']/(i+1))


st.write('Heads :', results['heads'])
st.write('Tails :', results['tails'])

fig = plt.figure()
plt.plot(prob)
# line colour is red
plt.axhline(y = 0.5, color = 'r', linestyle = 'dashed')  
# adding axis labels    
plt.xlabel('No. of Trails')
plt.ylabel('Probability of head')
st.pyplot(fig)




code = """
import random
import matplotlib.pyplot as plt 

results = {
    'heads': 0,
    'tails': 0,
}
sides = list(results.keys())
prob = []
for i in range(500):
    results[random.choice(sides)] += 1
    prob.append(results['heads']/(i+1))

print('Heads:', results['heads'])
print('Tails:', results['tails'])

plt.plot(prob)
# line colour is red
plt.axhline(y = 0.5, color = 'r', linestyle = 'dashed')  
# adding axis labels    
plt.xlabel('No. of Trails')
plt.ylabel('Probability of head')
plt.show()
"""
st.code(code, language='python')


st.markdown(r"""
#### The opposite of an event

The _opposite_ or _complement_ of an event $A$ is the event [not A] (that is, the event of A not occurring), often denoted as
${\displaystyle A',A^{c}}, {\displaystyle {\overline {A}},A^{\complement },\neg A}$ or ${\displaystyle {\sim }A}$
; its probability is given by 
$$
P(A^{c}) = 1 - P(A)
$$
As an example, the chance of not rolling a six on a six-sided die is 
$1 – (\textup{chance of rolling a six}) =1-{\tfrac {1}{6}}={\tfrac {5}{6}}$


#### Independent events
If two events, $A$ and $B$ are independent then the joint probability is
$$
{\displaystyle P(A{\textup{ and }}B)=P(A\cap B)=P(A)P(B).}
$$
For example, if two coins are flipped, then the chance of both being heads is ${\tfrac {1}{2}}\times {\tfrac {1}{2}}={\tfrac {1}{4}}$


#### Mutually exclusive events
If either event $A$ or event $B$ can occur but never both simultaneously, then they are called mutually exclusive events.

If two events are mutually exclusive, then the probability of both occurring is denoted as $P(A\cap B)$
and 
$$
{\displaystyle P(A{\textup{ and }}B)=P(A\cap B)=0}
$$
If two events are mutually exclusive, then the probability of either occurring is denoted as $P(A\cup B) $ and
$$
{\displaystyle P(A{\textup{ or }}B)=P(A\cup B)=P(A)+P(B)-P(A\cap B)=P(A)+P(B)-0=P(A)+P(B)}
$$
For example, the chance of rolling a 1 or 2 on a six-sided die is
$P(1{\textup{ or }}2)=P(1)+P(2)={\tfrac {1}{6}}+{\tfrac {1}{6}}={\tfrac {1}{3}}.$

#### Not mutually exclusive events
If the events are not mutually exclusive then
$$
{\displaystyle P\left(A{\textup{ or }}B\right)=P(A\cup B)=P\left(A\right)+P\left(B\right)-P\left(A{\textup{ and }}B\right).}
$$
For example, when drawing a card from a deck of cards, the chance of getting a heart or a face card (J,Q,K) (or both) is
${\tfrac {13}{52}}+{\tfrac {12}{52}}-{\tfrac {3}{52}}={\tfrac {11}{26}}$



#### Conditional probability
Conditional probability is the probability of some event $A$, given the occurrence of some other event $B$. Conditional probability is written
$P(A\mid B)$, , and is read "the probability of $A$, given $B$". It is defined by
$$
P(A\mid B)={\frac {P(A\cap B)}{P(B)}}.
$$
If $ P(B)=0$ then $P(A\mid B)$ is formally **undefined** by this expression. 
In this case $A$ and $B$ are independent,
 since ${\displaystyle P(A\cap B)=P(A)P(B)=0}$. 
""")
st.write('[Reference](https://en.wikipedia.org/wiki/Probability#:~:text=Probability%20is%20the%20branch%20of,event%20and%201%20indicates%20certainty.)')

st.markdown(r"""
### Bayes' theorem The opposite of an event
In probability theory and statistics, 
Bayes' theorem (alternatively Bayes' law or Bayes' rule; recently Bayes–Price theorem),
describes the probability of an event, based on prior knowledge of conditions that might be
 related to the event.


##### Statement of theorem
Bayes' theorem is stated mathematically as the following equation:
$$
{\displaystyle P(A\mid B)={\frac {P(B\mid A)P(A)}{P(B)}}}
$$
where $A$ and $B$ are events and ${\displaystyle P(B)\neq 0}$.

##### Naming the Terms in the Theorem
The terms in the Bayes Theorem equation are given names depending on the context where the equation is used.

- $P(A\mid B)$ is a conditional probability: the probability of event $A$ occurring given that $B$ is true. It is also called the **posterior probability** of $A$ given $B$.
- $P(B\mid A)$ is also a conditional probability: the probability of event $B$ occurring given that $A$ is true. It can also be interpreted as **the likelihood** of $A$ given a fixed $B$ because ${\displaystyle P(B\mid A)=L(A\mid B)}$.
- $P(A)$ and $P(B)$ are the probabilities of observing $A$ and $B$ respectively without any given conditions; they are known as **the marginal probability** or **prior probability**.

Sometimes $P(B)$ is referred to as **the evidence**. This allows Bayes Theorem to be restated as:
$$
\textup{Hello}
\textup{Posterior} = \frac{\textup{Likelihood} * \textup{Prior}}{\textup{Evidence}}
$$

##### False Positives and False Negatives
One of the famous uses for Bayes Theorem is False Positives and False Negatives. For those we have two possible cases for $A$, such as Pass/Fail (or Yes/No etc...)

###### Example: Allergy or Not?
There is a test for Allergy to Cats, but this test is not always right:
* For people that really do have the allergy, the test says "Yes" $80\%$ of the time.
* For people that do not have the allergy, the test says "Yes" $10\%$ of the time ("false positive").

If 1% of the population have the allergy, and your cat's test says "Yes", what are the chances that your cat really has the allergy?

So, we want to know the chance of having the allergy when test says "Yes", written $P(\textup{Allergy|Yes})$

Let's get our formula:
$$
P(\textup{Allergy|Yes}) = \frac{P(\textup{Yes|Allergy})P(\textup{Allergy})}{P(\textup{Yes})}
$$

* $P(\textup{Allergy})$ is Probability of Allergy = $1\%$
* $P(\textup{Yes|Allergy})$ is Probability of test saying "Yes" for people with allergy = $80\%$
* $P(\textup{Yes})$ is Probability of test saying "Yes" (to anyone) = ??$\%$

We don't know what the general chance of the test saying "Yes" is, but we can calculate it by 
adding up those **with**, and those **without** the allergy:

$$
P(\textup{Yes}) = P(\textup{Yes} \cap \textup{Allergy}) + P(\textup{Yes} \cap \textup{Not Allergy})\\
P(\textup{Yes}) = P(\textup{Yes | Allergy})P(\textup{Allergy}) + P(\textup{Yes | Not Allergy})P(\textup{Not Allergy})
$$

* $P(\textup{Yes|Not Allergy})$ is Probability of test saying "Yes" for people without allergy = $10\%$
* $P(\textup{Not Allergy})$ is Probability of Not Allergy = $99\%$

Let's add that up:
$$
P(\textup{Yes}) = 1\% \times 80\% + 99\% \times 10\% = 10,7\%
$$
Which means that about 10,7% of the population will get a "Yes" result.

So now we can complete our formula:
$$
P(\textup{Allergy|Yes}) = \frac{1\% \times 80\%}{10.7\%} = 7,48\%
$$

##### Special version of the Bayes' formula
After this example, we can write a special version of the Bayes' formula just for things like this:
$$
P(A|B) =  \frac{P(B|A) P(A)}{P(B|A) P(A) + P(B|A^c) P(A^c)}
$$


#### $A$ With Three (or more) Cases
We just saw $A$ with two cases ($A$ and not $A$), which we took care of in the bottom line.

When $A$ has 3 or more cases we include them all in the bottom line:

$$
P(A_1|B) =  \frac{P(B|A_1) P(A_1)}{P(B|A_1) P(A_1) + P(B|A_2) P(A_2) + P(B|A_3) P(A_3) + ...}
= \frac{P(B|A_1) P(A_1)}{\sum_{i=1}^n P(B|A_i) P(A_i)}
$$
""")


st.write("""
References :

* [Wikipedia](https://en.wikipedia.org/wiki/Bayes%27_theorem)
* [A Gentle Introduction to Bayes Theorem for Machine Learning](https://machinelearningmastery.com/bayes-theorem-for-machine-learning/)
* [Bayes' Theorem](https://www.mathsisfun.com/data/bayes-theorem.html)
""")

b1 = st.button('Youtube : Bayes Theorem - Example: A disjoint union')
b2 = st.button('Youtube : BBayes Theorem | Hate it or Love it, can t ignore it!')
if b1 :
    st.video("https://www.youtube.com/watch?v=k6Dw0on6NtM")
if b2:
    st.video("https://www.youtube.com/watch?v=bUI8ovd07uI")