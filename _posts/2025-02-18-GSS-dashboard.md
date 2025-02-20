---
layout: post
title: "GSS Dashboard"
image: "/posts/GSS_dashboard.png"
tags: [GSS, Data Cleaning, Visualization, Tableau]
---

## Summary & Context
This project demonstrates 1. Visualization using Tableau, and 2. Preprocessing and analyzing a large public dataset. 

I chose this project to practice Tableau as well as represent my background in the social sciences. It is also an effort to celebrate publically available data, and to make that data even more publically acessible. The General Social Survey, which recieves its primary funding from the National Science Foundation, has been collecting data on American political opinions and social behaviors since 1972. 

Coincidentally, while the analyses here tell a story of declining trust in American instutions alongside rising political polarization, current administration efforts to slash public spending puts funding for projects like the GSS in the crosshairs. 

## Visualization Using Tableau

<iframe seamless frameborder="0" src="https://public.tableau.com/views/GSS_2_17387981106750/GSSDashboard?:embed=yes&:display_count=yes&:showVizHome=no" width = '1100' height = '900'></iframe>

Unfortunately, I have difficulty right-sizing Tableau's dashboards for github pages. Automatic scaling in Tableau shifts objects in the view, and github's default width does not correspond with my desktop. 
Visit my Tableau Public Profile for a more responsive Dashboard
[50 Years of the General Social Survey](https://public.tableau.com/views/GSS_2_17387981106750/GSSDashboard?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

## The General Social Survey Dashboard

The General Social Survey is made of 34 waves of representative sample survey. The entire dataset contains over 70,000 rows and over 6000 columns. See the [GSS](https://gss.norc.org/) website for further details. 

My purpose here is not to provide extensive statistical or theoretical analysis of any one measure, but merely to offer a dashboard that allows uers to explore basic relationships through visualization. I constructed a series of measures at my discretion, including: 

* Education
* Health
* Happiness
* Social Attitudes
* Social Relationships
* Confidence in Institutions
* Confidence in Government
* Confidence in Media
* Work-Life Balance
* Quality of Life
* Religiosity
* Polarization

Click on a measure button to see its description, distribution, and trends. Toggle a demographic dimension. Select a specific year or all years. 

Details for each measure are shared below. Briefly, most measures are individual-level indices (expcept Education and Health, which are raw measures, and Polarization, which is an aggregation). All measures except Polarization are standardized for comparability. Global trend line data include imputed data for the sake of drawing a continuous trend line - but those imputed data do not carry over to other charts. 

Coding scripts are available in my GSS repository. Examples are shown below. 

### Measure and Dimension details

| **Variable Name** | **Variable Type** | **Description** | **GSS Variables** | 
|---|---|---|---|
| Education | Single Measure | Years of school, standardized. | EDUC |
| Health | Single Measure | Self reported Health, scale 1(poor health) - 4(excellent health), standardized. | HEALTH |
| Happiness | Index Measure | Average happiness, life excitement, and happy relationships, standardized. | HAPPY, LIFE, HAPMAR, HAPCOHAB |
| Religiosity | Index Measure | Average relgious service attendance, prayer, importance of religion, and strength of beliefs, standardized. | ATTEND, PRAY, RELITEN, GOD, BIBLE |
| Social Relationships | Index Measure | Average time spent with family, friends, neighbors, and out a bars, standardized. | SOCREL, SOCFREND, SOCOMMUN, SOCBAR |
| Social Attitudes | Index Measure | Average trust, and feelings about others being helpful and fair. | TRUST, HELPFUL, FAIR |
| Work-Life Balance | Index Measure | Product of job satisfaction and reversed hours/week worked for respondents with at least 10 hr/week, standardized. | SATJOB, HRS1 |
| Quality of Life | Index Measure | Average standardized Education, Health, Social Relationships, and Work-Life Balance. | EDUC, HEALTH, SOCREL, SOCFREND, SOCOMMUN, SOCBAR, SATJOB, HRS1 |
| Confidence in Institutions | Index Measure | Average confidence in 13 institutions. | CONEDUC, CONFED, CONMEDIC, CONARMY, CONBUS, CONCLERG, CONFINAN, CONJUDGE, CONLABOR, CONLEGIS, CONPRESS, CONSCI, CONTV |
| Confidence in Government | Index Measure | Average confidence in congress and executive branch. | CONFED, CONLEGIS |
| Confidence in Media | Index Measure | Average confidence in television and the press. | CONPRESS, CONTV |
| Age Group | Dimension | Respondent' age categorized for comparability. | AGE |
| Degree | Dimension | Respondent's education level (note correlation with Education, above). | DEGREE |
| Race | Dimension | Respondent's race. | RACE |
| Socioeconomic Status | Index Dimension | Standardized education (years of school), occupational prestige, and total family income, split: 30%(low) / 40%(mid) / 30%(high). | EDUC, PRESTG10, REALINC |
| Gender | Dimension | Repondent's gender. | SEX |
| Race * Gender | Dimension | Repondent's Race/Gender. | RACE, SEX |
| Race * SES | Dimension | Respondent's Race/Socioeconomic status. | RACE, EDUC, REALINC, PRESTG10 |
| Place | Dimension | Type of Social Environment, recodes using SIZE and XNORCSIZ. | SIZE, XNORCSIZ |
| Region | Dimension | Region of interview. All nine categories are used in the map. Collapsed to four categories for dimension comparison. | REGION |
___

## Sample Processing, Recoding & Index Construction
In the code below, I will
* Select variables from GSS for the dashboard.
* Demonstrate key variable recodes. 
* Construct indices, demonstrating reliability analysis, factor analysis, and a minimum variable threshold.
* Impute measures for continuous trend lines.
<br>

### Select Variables

```python
# import required packages and libraries
import os 
import pyreadstat # reads sas files
import pandas as pd
import numpy as np
import pingouin as pg
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler # I'll also write my own standardization function. 
import matplotlib.pyplot as plt

os.chdir("C:/Dat_Sci/Data Projects/GSS/dashboard project")

df, meta = pyreadstat.read_sas7bdat('C:/Dat_Sci/Datasets/GSS_sas/gss7222_r3.sas7bdat')

df.shape
Out[5]: (72390, 6691)
```

6,691 is a lot of columns. The only way to figure out which columns are of interest is to go to the GSS website. They have a variable explorer dashboard, which allows you search variables by topic area, or by module, including their "core" module - measures that are collected at every wave, and many others. The following are selected after extensive digging, researching and iteratively realizing what I want to look at. 

```python
trends = df[["YEAR", "SIZE", "XNORCSIZ", "AGE", "SEX", "EDUC", "PRESTG10", "REALINC",
           "DEGREE", "RACE", "HAPPY", "TRUST", "HELPFUL", "FAIR", "HEALTH", "LIFE", 
           "HAPMAR", "HAPCOHAB", "RELITEN", "GOD", "BIBLE", "REGION", "ATTEND", 
           "PRAY", "HRS1", "SATJOB", "MOBILE16", "POLVIEWS", "PARTYID", "SOCREL", 
           "SOCOMMUN", "SOCBAR", "SOCFREND", "CONEDUC", "CONFED", "CONMEDIC", "CONARMY",
           "CONBUS", "CONCLERG", "CONFINAN", "CONJUDGE", "CONLABOR", 
           "CONLEGIS", "CONPRESS", "CONSCI", "CONTV"]].copy()
```

### Key Variable Recodes

I demonstrate recodes and index construction for two key measures: Confidence in Institutions & Work-Life Balance

```python

## Confidence in Institutions ##

## GSS Script: "I am going to name some institutions in this country. As far as the people running these institutions are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them?"
## GSS Codes:
# 1: "a great deal"
# 2: "only some"
# 3: "hardly any"

# I'll recode them in a positive direction so that outcomes are easily interpretatble: high numbers = high confidence. All at once using a list.

conf_vars = ["CONEDUC", "CONFED", "CONMEDIC", "CONARMY", "CONBUS", 
             "CONCLERG", "CONFINAN", "CONJUDGE", "CONLABOR",
             "CONLEGIS", "CONPRESS", "CONSCI", "CONTV"]

for var in conf_vars: 
    trends[var.lower()] = 4 - trends[var] # subtracting 4 from the original yeilds the reverse order. 

# I change the case to keep the variable name while indicating it has been recoded. And then drop the original. 
trends.drop(conf_vars, axis = 1, inplace=True)

# Now construct the Index
conf_vars = [var.lower() for var in conf_vars] # First I want the same list in lower case because I renamed the variables to lower case.
trends["conf_index"] = trends[conf_vars].mean(axis = 1, skipna = True) # take the mean, instead of the sum, so that rows with missing values don't bring down the totals.

trends["conf_index"].describe()
Out[12]: 
count    48957.000000
mean         2.031674
std          0.352033
min          1.000000
25%          1.800000
50%          2.000000
75%          2.230769
max          3.000000
Name: conf_index, dtype: float64

Now we have an index, but how well to the component measures cohere, and how to the respondents' answers cohere?

# Alpha Analysis for inter-item reliability

confidence_columns = trends[conf_vars].copy() # set the component columns apart

reliability = pg.cronbach_alpha(data = confidence_columns)
print(reliability) # Alpha = .79
# An alpha of .79 indicates high inter-item reliability. The of the measure are all getting after one underlying construct, confidence in institutions. 

# B. Factor Analysis 

fa = FactorAnalyzer(n_factors=3, rotation="varimax")  # Adjust n_factors as needed
fa.fit(confidence_columns)

# Get eigenvalues to determine the optimal number of factors
ev, v = fa.get_eigenvalues()
print("Eigenvalues:", ev)
Eigenvalues: [3.54629544 1.41743876 1.00764509 0.96894189 0.86242839 0.80770145
 0.75664672 0.70908479 0.67070858 0.61520566 0.58425706 0.55188097
 0.50176519]
# eigenvalues > 1 indicate a useful factor. In this case there are 3 strong factors

# Print factor loadings
loadings = fa.loadings_
print("Factor Loadings:\n", loadings)
Factor Loadings:
 [[ 0.36118129  0.24410226  0.23097058]
 [ 0.19306445  0.59500797  0.1643183 ]
 [ 0.62462896 -0.00825083  0.17924858]
 [ 0.45896996  0.17788212 -0.03423817]
 [ 0.41959102  0.25217145  0.09374704]
 [ 0.28677819  0.22292182  0.11038094]
 [ 0.38402004  0.25974974  0.17509192]
 [ 0.45309956  0.3305685   0.10799279]
 [ 0.06857161  0.2834985   0.27623394]
 [ 0.16888011  0.6249543   0.29079604]
 [ 0.09311359  0.15853253  0.66356791]
 [ 0.52642256  0.03028766  0.04476395]
 [ 0.1125664   0.16761342  0.54758137]]
# factor loadings > .4 indicate a useful inclusion

# next step: separate and name three domains 
factor1 = ["conmedic", "conarmy", "conbus", "confinan", "conjudge"] # generic confidence in institutions
factor2 = ["confed", "conlegis"] # confidence in the government
factor3 = ["conpress", "contv"] # confidence in the media

# subsequent reliability analysis does not shown an improvement of factor1 (a subset of 5) over the total index measuring confidence in 13 institutions.
# For the dashboard, I'll keep the Confidence in Institutions Index, and I'll add Confidence in Government and Confidence in Media Indices for comparison.
```
<br>

#### Variables Sensitivity Test
Earlier I mentioned that taking the mean of the component measures is better than summing to the get the index because if there are missing values in some of the component measures, then a respondent's total index score is artificially lowered. It's true that taking the mean of any number of variables of range 1-3 stil yeilds a range of 1-3, which doesn't feel like a composite index, but we would standardize in the end anyway, which will (ideally) yeild us a zero-centered range of -3 to 3 either way. But when we do take the mean, we should be careful that the output represents the same thing for all respondents. If one respondent answered a 3 for one measure and then refused to continue, that would be much different than a respondent who answered carefully for all 13 measures. We can count the number of items a respondent answered, and compare means, distributions, and n for respondents that answered n, n-1, n-2, etc. 

```python
trends["num_conf_vars"] = trends[confidence_vars].notna().sum(axis=1) # count the number of missing items per row. 

thresholds = [6, 8, 10, 11, 12] # I'll compare distributions for each minimum number of questions answered. 
results = {} 
for threshold in thresholds:
    temp_df = indexes.copy() # Create a copy of the dataset to avoid modifying the original
    # Recalculate the confidence index based on the threshold
    temp_df["conf_index"] = temp_df[confidence_vars].mean(axis=1, skipna=True)
    temp_df.loc[temp_df["num_conf_vars"] < threshold, "conf_index"] = np.nan
    
    # Summarize the results
    results[threshold] = temp_df["conf_index"].describe()

# Display the summaries for each threshold
for threshold, summary in results.items():
    print(f"Threshold: {threshold}")
    print(summary)
    print("\n" + "="*50 + "\n")

Threshold: 6
count    48665.000000
mean         2.031716
std          0.350796
min          1.000000
25%          1.800000
50%          2.000000
75%          2.230769
max          3.000000
Name: conf_index, dtype: float64

==================================================

Threshold: 8
count    48290.000000
mean         2.031468
std          0.350350
min          1.000000
25%          1.800000
50%          2.000000
75%          2.230769
max          3.000000
Name: conf_index, dtype: float64

==================================================

Threshold: 10
count    47510.000000
mean         2.031037
std          0.349782
min          1.000000
25%          1.800000
50%          2.000000
75%          2.230769
max          3.000000
Name: conf_index, dtype: float64

==================================================

Threshold: 11
count    46674.000000
mean         2.030616
std          0.349311
min          1.000000
25%          1.818182
50%          2.000000
75%          2.230769
max          3.000000
Name: conf_index, dtype: float64

==================================================

Threshold: 12
count    44891.000000
mean         2.029900
std          0.348355
min          1.000000
25%          1.769231
50%          2.000000
75%          2.230769
max          3.000000
Name: conf_index, dtype: float64

==================================================
# Fortunately, there is very little change in n, and amost no change in the mean. Using a minimum of 8 variables for the index seems to provide the best trade off. 
# Use minimum 8 variables
# trends["conf_index"] = trends[confidence_vars].mean(axis=1, skipna=True) # index construction again
trends.loc[trends["num_conf_vars"] < 8, "conf_index"] = np.nan # calling all rows with less than 8 answers "nan." 
```


####  Standardize
Because I'll end up with a dozen or so index measures, they'll all be standardized for easy comparison. One way is to use scikitlearn's StandardScaler(), another is to simply subtract the mean and divide by the standard deviation.

```python

def standardize(x):
    return (x - x.mean()) / x.std()

dashboard_measures = ["education", "health", "religiosity", "social_attitudes", "social_relationships", "work_life_balance", "quality_of_life, conf_index", "conf_gov", "conf_media"]

for i in dashboard_measures:
    trends[i + "_z"] = standardize(trends[i]) 
```

### Tableau 
