---
title: "A Group-Stage Data Analysis of World Cup 2018"
layout: post
date: 2018-09-11
tag:
- World Cup 2018
- FIFA
- Group-stage
- Knockout-stage
- A Draw
headerImage: true
projects: true
hidden: true
description: "A Group-Stage Data Analysis of World Cup 2018"
category: project
author: leealbert
externalLink: false
---

# Executive Summary

World Cup 2018, the soccer tournament held every four years by FIFA, took place in Russia from June 14 to July 15, 2018.  Thirty two teams from all over the world participated in the tournament. They were divided into eight groups with four teams in each group. A total of sixty-four games had been played.  Of the 64 games, 48 games were played in the group-stage. The top two teams with the highest points in each group advanced to the knockout-stage.  In this data analysis we attempt to answer the following two questions:

1. What is the chance for two teams to play to a draw in the group-stage? Is it different from the chance of a draw in the knockout-stage?

2. There are three outcomes when two teams meet in the group-stage: the winning team receives 3 points; a draw is a point for each team; and the losing team gets zero point.  Why not give 2 points to the winning teams, instead of 3 points? Can a team win no games in the group-stage advance to the knockout-stage?  


# Introduction

In the group-stage, every team plays three games within their group. For instance, Russia, Uruguay, Egypt and Saudi Arabia were in Group A. Every team plays 3 games and there are 4 teams, so the total number of games should be 12. Since Russia and Uruguay and Uruguay and Russia will play only once. So the total number games played within the group is actually only $\binom{4}{2} = 6$ games.  There are eight groups.  So the total number of games in group-stage is $6 * 8 = 48$ games. This is the sample space from which we are to estimate the chance of a draw.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

wc = pd.read_csv('worldcup_2018.csv')
```


```python
wc.info
```




    <bound method DataFrame.info of    group       country  match outcome  points  goals
    0      A         Egypt      1       L       0      0
    1      A         Egypt      2       L       0      1
    2      A         Egypt      3       L       0      1
    3      A        Russia      1       W       3      5
    4      A        Russia      2       W       3      3
    5      A        Russia      3       L       0      0
    6      A  Saudi Arabia      1       L       0      0
    7      A  Saudi Arabia      2       L       0      0
    8      A  Saudi Arabia      3       W       3      2
    9      A       Uruguay      1       W       3      1
    10     A       Uruguay      2       W       3      1
    11     A       Uruguay      3       W       3      3
    12     B          Iran      1       W       3      1
    13     B          Iran      2       L       0      0
    14     B          Iran      3       D       1      1
    15     B       Morocco      1       L       0      0
    16     B       Morocco      2       L       0      0
    17     B       Morocco      3       D       1      2
    18     B      Portugal      1       D       1      3
    19     B      Portugal      2       W       3      1
    20     B      Portugal      3       D       1      1
    21     B         Spain      1       D       1      3
    22     B         Spain      2       W       3      1
    23     B         Spain      3       D       1      2
    24     C     Australia      1       L       0      1
    25     C     Australia      2       D       1      1
    26     C     Australia      3       L       0      0
    27     C       Denmark      1       W       3      1
    28     C       Denmark      2       D       1      1
    29     C       Denmark      3       D       1      0
    ..   ...           ...    ...     ...     ...    ...
    66     F   South Korea      1       L       0      0
    67     F   South Korea      2       L       0      1
    68     F   South Korea      3       W       3      2
    69     F        Sweden      1       W       3      1
    70     F        Sweden      2       L       0      1
    71     F        Sweden      3       W       3      3
    72     G       Belgium      1       W       3      3
    73     G       Belgium      2       W       3      5
    74     G       Belgium      3       W       3      1
    75     G       England      1       W       3      2
    76     G       England      2       W       3      6
    77     G       England      3       L       0      0
    78     G        Panama      1       L       0      0
    79     G        Panama      2       L       0      1
    80     G        Panama      3       L       0      1
    81     G       Tunisia      1       L       0      1
    82     G       Tunisia      2       L       0      2
    83     G       Tunisia      3       W       3      2
    84     H      Columbia      1       L       0      1
    85     H      Columbia      2       W       3      3
    86     H      Columbia      3       W       3      1
    87     H         Japan      1       W       3      2
    88     H         Japan      2       D       1      2
    89     H         Japan      3       L       0      0
    90     H        Poland      1       L       0      1
    91     H        Poland      2       L       0      0
    92     H        Poland      3       W       3      1
    93     H       Senegal      1       W       3      2
    94     H       Senegal      2       D       1      2
    95     H       Senegal      3       L       0      0

    [96 rows x 6 columns]>



# The Dataset

There are 96 rows of records representing three matches for each of the 32 teams in the group-stage.

# Group-Stage


```python
mtch1 = wc[wc.match==1]
```


```python
plt.hist(mtch1['outcome'])
```




    (array([ 6.,  0.,  0.,  0.,  0., 13.,  0.,  0.,  0., 13.]),
     array([0. , 0.2, 0.4, 0.6, 0.8, 1. , 1.2, 1.4, 1.6, 1.8, 2. ]),
     <a list of 10 Patch objects>)




![png](output_8_1.png)


Of the 16 games played on Match Day 1, three games resulted in a draw or 6 teams received one point.


```python
mtch2 = wc[wc.match==2]
plt.hist(mtch2['outcome'])
```




    (array([ 4.,  0.,  0.,  0.,  0., 14.,  0.,  0.,  0., 14.]),
     array([0. , 0.2, 0.4, 0.6, 0.8, 1. , 1.2, 1.4, 1.6, 1.8, 2. ]),
     <a list of 10 Patch objects>)




![png](output_10_1.png)


Of the 16 games played on Match Day 2, 2 games resulted in a draw or 4 teams received 1 point.


```python
mtch3 = wc[wc.match==3]
plt.hist(mtch3['outcome'])
```




    (array([ 8.,  0.,  0.,  0.,  0., 12.,  0.,  0.,  0., 12.]),
     array([0. , 0.2, 0.4, 0.6, 0.8, 1. , 1.2, 1.4, 1.6, 1.8, 2. ]),
     <a list of 10 Patch objects>)




![png](output_12_1.png)


Of the 16 games played on Match Day 3 four games resulted in a draw or eight teams received 1 point.  

We have three estimates of the chance of a draw from each Match Day in the group-stage.  They are $\frac{6}{32} = 0.1875$ for Match Day 1, $\frac{4}{32} = 0.125$ for Match Day 2 and $\frac{8}{32} = 0.25$ for Match Day 3.  We can also estimate the chance of a draw using the overall average $\frac{18}{96} = 0.1875$.


```python
plt.hist(wc['outcome'])
```




    (array([18.,  0.,  0.,  0.,  0., 39.,  0.,  0.,  0., 39.]),
     array([0. , 0.2, 0.4, 0.6, 0.8, 1. , 1.2, 1.4, 1.6, 1.8, 2. ]),
     <a list of 10 Patch objects>)




![png](output_14_1.png)


## Hypothesis testing - proportion z-test

If we assume equally likely probability for a game to result in a win, a draw or a loss the probability of a draw is estimated at $\frac{1}{3}$ or 0.33. We can test if the probability of a draw in group-stage is 0.33 when we just observed 9 draws in 48 games using the codes below:


```python
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

count = 9
nobs = 48
value = .33
stat, pval = proportions_ztest(count, nobs, value)
print('{0:0.3f}'.format(pval))
```

    0.011


## Result_1

The p-value of the proportion z-test is 0.011, which is less than 0.05. We reject the null hypothsis that the chance of a draw in the group-stage is 0.33.  The sample size $n = 48$ is reasonably large. The probability of a draw in group-stage is likely to be smaller than one-thirds.

# Knockout-Stage

In the subsequent knockout-stage there is no draw outcome *per se*.  A game can result in a draw before and after the extra time.  We want to compare the chance of a draw in the knockout-stage to the probability of a draw in the group-stage.

The top two teams with highest points within the group advance to the knockout-stage.  Sixteen teams advanced to the knockout-stage.  A total of 8 games was played.  In the knockout-stage if two teams play to a draw in regulation two 15-minutes extra time will be added to determine a winner.  After the extra time if a winner still can't be decided penalty kick will eventually determine a winner.  Of the 8 games played three games resulted in a draw.  The chance of a draw in the knockout-stage is hence estimated to be 3 out of 8 or 0.375.

If we make the same assumption that the chance of a draw in knockout-stage is 1 out of 3 or .333. We can perform a proportion test.  However, the sample size is rather small.

#this code does not work
import scipy.stats as stats
import plotly.plotly as py
from plotly.tools import FigureFactory as FF

data1 = [1,1,1,0,0,0,0,0]

true_mu = 0.33

onesample_results = scipy.stats.ttest_1samp(data1, true_mu)

matrix_onesample = [
    ['', 'Test Statistic', 'p-value'],
    ['Sample Data', onesample_results[0], onesample_results[1]]
]

onesample_table = FF.create_table(matrix_onesample, index=True)
py.iplot(onesample_table, filename='onesample-table')


```python
import scipy
import scipy.stats as stats

data1 = [1,1,1,0,0,0,0,0]

true_mu = 0.33

onesample_results = scipy.stats.ttest_1samp(data1, true_mu)
onesample_results
```




    Ttest_1sampResult(statistic=0.24592681838303027, pvalue=0.8127940492693482)



## Result_2

The p-value of this one sample t-test is 0.8128, which is greater than 0.05 so we do not reject the null hypothesis that the chance of a draw in the knockout-stage is 0.33.  

## Hypothese testing - Odds Ratio Test

One can also use odds ratio of 3 draws to 5 non-draws in knockout-stage to compare to odds ratio of 9 draws to 39 non-draws in the group-stage. Using a small sample approach, it is 2.6 times as likely to play to a draw in the knockout-stage than in the group-stage. However, it is not statistically significant at 0.95 level, due to small sample sizes.


```python
n = np.array([[3,5],[9,39]])

oddsratio, pvalue = stats.fisher_exact(n)
print("oddsR: ", oddsratio, "p-value: ", pvalue)
```

    oddsR:  2.6 p-value:  0.34829418559876724


## Result_3

The p-value of the odds ratio using Fisher's Exact test is 0.3483. It is not statistically significant at 0.05 level.

It is actually more likely that a game ends in a draw during the knockout-stage.  Teams tend to play very cautiously during the knockout-stage, which is often makes for pretty low scoring games as teams know they have extra time available and, in worst case scenario, using penalty kicks to advance.

As far as strategy goes, stronger teams should always play to get a win in group-stage and weaker teams should always play to avoid a loss.  Once the weaker teams make it out of the group-stage a strategy to help them to get to the semifinal would be to play to a draw and force a penalty kick - the World Cup 2018 data seem to support this argument - it is more likely to play to a draw in the knockout-stage than in the group-stage.

# Can a team advance to knockout-stage with 3 Draws

Here we will discuss the chance for a team to advance to the knockout-stage without a win (and without a loss) using 3-point system.


```python
import numpy as np
import pandas as pd
from collections import OrderedDict
```


```python
list_tuples = [(0,1),(1,3),(2,3),(3,4),(4,6),(5,3),(6,3),(7,3),(8,0),(9,1)]

val, freq = np.array(list_tuples).T
```


```python
def mean_(val, freq):
    return np.average(val, weights = freq)

def median_(val, freq):
    ord = np.argsort(val)
    cdf = np.cumsum(freq[ord])
    return val[ord][np.searchsorted(cdf[-1] // 2, cdf)]
```


```python
mean_(val, freq)
```




    4.0



#this would not work
median_(val, freq)

#comparison of theoretical and empirical distributions using 3-point system
#some weird thing happened here when I did this subplot

fig=plt.figure(figsize= [12,12])
ax1 = fig.add_subplot(111)
ax1.bar(df3['Points'],df3['Probability']);

ax2 = fig.add_subplot(212)
ax2.bar(empirical3['Points'],empirical3['Probability']);

fig.suptitle('Comparison of theoretical and Empirical')
plt.subplots_adjust(hspace = 2)

plt.show()

#comparison of theoretical and empirical distributions using 2-point system
#similar thing happened as in the previous subplot

fig=plt.figure(figsize= [12,12])
ax1 = fig.add_subplot(111)
ax1.bar(df2['Points'],df2['Probability']);

ax2 = fig.add_subplot(212)
ax2.bar(empirical2['Points'],empirical2['Probability']);

fig.suptitle('Comparison of theoretical and Empirical - 2-Point')
plt.subplots_adjust(hspace = 1)

plt.show()

## Probability Distribution of Points in 2-point System


```python
#theoretical distribution for 2-point system

point2 = [(0,1), (1,3), (2,6), (3,7), (4,6), (5,3), (6,1)]

x2 = sum(([val,]*freq for val, freq in point2),[])

np.median(x2),np.mean(x2)
```




    (3.0, 3.0)




```python
record2 = OrderedDict([ ('Points', [0,1,2,3,4,5,6]),
          ('Probability', [1/27, 3/27, 6/27, 7/27, 6/27, 3/27, 1/27])]
           )
df2 = pd.DataFrame.from_dict(record2)
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Points</th>
      <th>Probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.037037</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.111111</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.222222</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.259259</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.222222</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.111111</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.037037</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.bar(df2['Points'],df2['Probability'] )
```




    <Container object of 7 artists>




![png](output_37_1.png)


## Empirical Distribution of Points in 2-point System


```python
#empirical distribution for 2-point system

sample2 = [(0,2), (1,4), (2,8), (3,4), (4,9), (5,2), (6,3)]

x21 = sum(([val,]*freq for val, freq in sample2),[])

np.median(x21),np.mean(x21)
```




    (3.0, 3.0)




```python
sample21 = OrderedDict([ ('Points', [0,1,2,3,4,5,6]),
          ('Probability', [2/32, 4/32, 8/32, 4/32, 9/32, 2/32, 3/32])]
           )
empirical2 = pd.DataFrame.from_dict(sample21)
empirical2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Points</th>
      <th>Probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.06250</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.12500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.25000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.12500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.28125</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.06250</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.09375</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.bar(empirical2['Points'],empirical2['Probability'])
```




    <Container object of 7 artists>




![png](output_41_1.png)


## Probability Distribution of points in 3-point System


```python
#theoretical distribution for 3-point

point3 = [(0,1), (1,3), (2,3), (3,4), (4,6), (5,3), (6,3), (7,3), (9,1)]

x3 = sum(([val,]*freq for val, freq in point3),[])

np.median(x3),np.mean(x3)
```




    (4.0, 4.0)




```python
record3 = OrderedDict([('Points', [0,1,2,3,4,5,6,7,8,9]),
          ('Probability', [1/27, 3/27, 3/27, 4/27, 6/27, 3/27, 3/27, 3/27, 0/27, 1/27])]
           )
df3 = pd.DataFrame.from_dict(record3)
df3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Points</th>
      <th>Probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.037037</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.111111</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.111111</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.148148</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.222222</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.111111</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.111111</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.111111</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0.037037</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.bar(df3['Points'],df3['Probability'])
```




    <Container object of 10 artists>




![png](output_45_1.png)


## Empirical Distribution of points in 3-point System


```python
#empirical distribution for 3-point

sample3 = [(0,2), (1,4), (2,0), (3,8), (4,4), (5,4), (6,5), (7,2), (9,3)]

x31 = sum(([val,]*freq for val, freq in sample3),[])

np.median(x31),np.mean(x31)
```




    (4.0, 4.21875)




```python
sample31 = OrderedDict([ ('Points', [0,1,2,3,4,5,6,7,8,9]),
          ('Probability', [2/32, 4/32, 0/32, 8/32, 4/32, 3/32, 4/32, 2/32, 0/32, 3/32])]
           )
empirical3 = pd.DataFrame.from_dict(sample31)
empirical3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Points</th>
      <th>Probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.06250</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.12500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.25000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.12500</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.09375</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.12500</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.06250</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0.09375</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.bar(empirical3['Points'],empirical3['Probability'])
```




    <Container object of 10 artists>




![png](output_49_1.png)


As you can see from the two graphs: using 2-point system the graph is symmetric. So the mean and median are the same at 3 points. A team with 3 points in 2-point system can be three draws or one win, one draw and one loss and still qualify for knockout-stage. In World Cup 2018 there was no team with three draws.  Japan and Senegal are the only two teams with the same recod of a win, a draw and a loss. Japan eventually advanced using one of rarely used tie-breaker.

Using 3-point system the distribution is skewed to the right.  However, the mean and the median happen to be the same at 4 points.  A team with three draws in the 3-point system will not qualify.  In order to qualify in the 3-point you have to have a win and at least one draw like that of Japan and Senegal in World Cup 2018. Even with 4 points like Senegal, advancing is not guaranteed, either.

# Conclusion

World Cup 2018 has many dramatic moments. Few, if any, expected Russia, the hosting country, to advance to the  knockout-stage.  Not only did Russia made it to the round of 16, but also beat the soccer powerhouse Spain in a penalty kick, only to lose to Croatia in the penalty kick (3-4).

Germany, the reigning world cup champion in 2014, did not make it out of the group-stage; Soccer powerhouses Spain (2010 World Cup champion), Portugal, Argentina were all eliminated before the quarter-finals. Brazil, favored to win it all before the tournament, lost to Belgium in the quarter-final;

England blew away Panama (6-1) on Match Day 2. An investment banking company, Goldman Sachs, has used a machine learning algorithm to predict England will reach the World Cup final. The algorithm simulated a million variations to pick a winner ahead of the tournament. The pre-tournament forecasted a final between Germany and Brazil.

After Match Day 2 Goldman Sachs now predicts that England will meet Brazil in the final. It was not the case at all, as England was beaten by Croatia in the semi-final.

Japan and Senegal ended up with the same record and a rarely used tie-breaker was applied to advance Japan to the knockout-stage; only to lose to Belgium, who came from 2 goals behind in the final minutes of regulation;

Denmark was the only team with back-to-back draw and it occurred on the Match Day 2 and 3. There was no team with three draws in group-stage.

Croatia, ranked 20th in the FIFA World, was one of three teams with three wins in the group-stage (Uruguay and Belgium are the other two countries); Croatia also had to play three consecutive extra time games: beat Denmark in penalty kick (3-2) in the round of 16; beat Russia in penalty kick (4-3) in the quarter-final and beat England in extra time 2-1, to get to the final to face France.

Twenty of the 32 teams appeared in World Cup 2014, France was one of them.  France, ranked 7th in the FIFA World, won 2 games and a draw with a total of 7 points in the group-stage; they shut out Uruguay 2-0 in the quarter-final and beat Belgium 1-0 in the semi-final and win it all by beating Croatia 4-2 in the final.

The chance of three draws in group-stage is very rare.  Under the 3-point system a team with three draws will not qualify for knockout-stage. To qualify you have to have at least one win and no loss. Under the 2-point system three points are right at the median. A team with three draws could advance to the knockout-stage in 2-point system.  Maybe this is the reason that FIFA implemented the 3-point system long time ago.
