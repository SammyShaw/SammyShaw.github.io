---
layout: post
title: "GSS Dashboard"
image: "/posts/GSS_dashboard.png"
tags: [GSS, Data Cleaning, Visualization, Tableau]
---

## Summary
This project demonstrates 1. Visualization using Tableau, and 2. Getting and preprocessing a large public dataset. 

This project 



## Visualization Using Tableau

<iframe seamless frameborder="0" src="https://public.tableau.com/views/GSS_2_17387981106750/GSSDashboard?:embed=yes&:display_count=yes&:showVizHome=no" width = '1100' height = '900'></iframe>

Visit my Tableau Public Profile for a more responsive Dashboard
[50 Years of the General Social Survey](https://public.tableau.com/views/GSS_2_17387981106750/GSSDashboard?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

## The General Social Survey

The General Social Survey is a publically available data set made of 34 waves of representative sample data since 1972. The entire dataset contains over 70,000 rows and over 6000 columns. See the [GSS](https://gss.norc.org/) website for further details. 

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
