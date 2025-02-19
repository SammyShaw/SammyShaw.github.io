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
|---|---|---|---|---|
| Education | Single Measure | Years of school, standardized. | EDUC |  | 
| Health | Single Measure | Self reported Health, scale 1(poor health) - 4(excellent health), standardized. | HEALTH |
| Happiness | Index Measure | Average happiness, life excitement, and happy relationships, standardized. | HAPPY, LIFE, HAPMAR, HAPCOHAB |
| Religiosity | Index Measure | Average relgious service attendance, prayer, importance of religion, and strength of beliefs, standardized. | ATTEND, PRAY, RELITEN, GOD, BIBLE |
| Social Relationships | Index Measure | Average time spent with family, friends, neighbors, and out a bars, standardized. | SOCREL, SOCFREND, SOCOMMUN, SOCBAR |
| Social Attitudes | Index Measure | Average trust, and feelings about others being helpful and fair | TRUST, HELPFUL, FAIR |
| Work-Life Balance | Index Measure | Product of job satisfaction and reversed hours/week worked for respondents with at least 10 hr/week, standardized. | SATJOB, HRS1 |
| Quality of Life | Index Measure | Average standardized Education, Health, Social Relationships, and Work-Life Balance | EDUC, HEALTH, SOCREL, SOCFREND, SOCOMMUN, SOCBAR, SATJOB, HRS1 |
| Confidence in Institutions | Index Measure | Average confidence in 13 institutions | CONEDUC, CONFED, CONMEDIC, CONARMY, CONBUS, CONCLERG, CONFINAN, CONJUDGE, CONLABOR, CONLEGIS, CONPRESS, CONSCI, CONTV |
| Confidence in Government | Index Measure | Average confidence in congress and executive branch | CONFED, CONLEGIS |
| Confidence in Media | Index Measure | Average confidence in television and the press | CONPRESS, CONTV |
| Age Group | Dimension | Respondent' age categorized for comparability | AGE |
| Degree | Dimension | Respondent's education level (note correlation with Education, above) | DEGREE |
| Race | Dimension | Respondent's race | RACE |
| Socioeconomic Status | Index Dimension | Standardized education (years of school), occupational prestige, and total family income, split 30(low)/40(mid)/30(high)% | EDUC, PRESTG10, REALINC |
| Gender | Dimension | Repondent's gender | SEX |
| Race * Gender | Dimension | Repondent's Race/Gender | RACE, SEX |
| Race * SES | Dimension | Respondent's Race/Socioeconomic status | RACE, EDUC, REALINC, PRESTG10 |
| Place | Dimension | Type of Social Environment, recodes using SIZE and XNORCSIZ | SIZE, XNORCSIZ |
| Region | Dimension | Region of interview. All nine categories are used in the map. Collapsed to four categories for dimension comparison | REGION |
___


### Sample 
