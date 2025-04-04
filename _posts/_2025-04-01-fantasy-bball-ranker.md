---
layout: post
title: Fantasy BBall Ranking using SHAW-Transformation
image: "/posts/PLACEHOLDER.png"
tags: [ETL Pipeline, Statistics, System Ranking, Python, Fantasy Basketball]
---

In this project I adresses the percentage distribution problem in fantasy basketball ranking algorithms. Existing ranking systems weight percentage performances linearly by attempts. Because percentages are bound between 0 and 1, however, the actual affect of attempts on percentages is asymptotic, not linear. I develop a Sigmoid-Harmonic-Attempt-Weighting (SHAW) transformation tied to the statistical properties (CoV & Skewness) of attempt distributions. I then use the SHAW percentage transformations to create six unique fantasy basketball ranking algorithms which I then systematically compare to the current leading fantasy platform rankings: ESPN, Yahoo.com, and Basketball Monster. In top-100 player head-to-head matchups, two of my ranking systems beat Basketball Monster, three beat Yahoo, and all six beat ESPN. I extract, transform, and load data for my own fantasy basketball player ranking app. [URL]. 

## Contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Literature Review](#literature-review)
    - [Actions](#overview-actions)
    - [Results & Discussion](#overview-results)
- [01. Extract: Data Overview](#extract)
- [02. Transform: Goals](#transformation-overview)
    - [Distributions & Redistributions](#distributions)
    - [Percentages](#percentages)
    - [SHAW-Transformation](#shaw-tranformation)
    - [Standardization](#standardization)
    - [Min-Max Normalization](#min-max)
    - [Scarcity Index](#scarcity)
    - [Sum of Category Ranks](#sum-category-ranks)
    - [H2H Player Matchup](#H2H-matchups)
- [03. Compare: Ranking Matchups](#ranking-matchups)
    - [Top-N players](#top-n)
    - [Snake-Draft Ranking Tournament](#tournament)
- [04. VORP](#value-over-replacement)
- [05. Load: Streamlit App](#streamlit-app)
- [06. Discussion & Conclusion](#discussion)


# Project Overview

### Context

Fantasy sports are a popular past time with over 62.5 million participants in the U.S. and Canada [FSGA.org], 20 million of whom play fantasy basketball. In a fantasy sports league, participants (or, team managers) create a fantasy team by competitively drafting from a pool of professional players, whose real stats become their own fantasy stats for the season. Choosing a successful team requires knowing the players and understanding the league’s scoring format. In *9-category* fantasy basketball leagues (the standard competitive fantasy basketball format), teams are compared – and thus players must be evaluated - across nine different measures. To make this easier, the major platforms Yahoo.com and ESPN include player rankings to help managers make their choices. But Yahoo and ESPN often include questionable players in their top ranks, and unfortunately, neither platform is transparent about how their rankings are calculated. As a regular fantasy basketballer, I’ve often thought that these rankings could be improved. 

### Literature Review

I’m not the only one to question the platform rankings and there are some interesting contributions out there concerning the best way to construct them. The consensus in these discussions is that the existing (Yahoo and ESPN) rankings are made by standardizing the categories and then ranking the sum of the players’ *Z*-Scores (with caveats for percentage categories). But standardization is criticized for over- or under-valuing outliers if distributions are skewed (which they are in the NBA). Victor Wembanyama’s *Z*-score of 8 in the Blocks category, for example, statistically represents a nearly impossible occurrence when assuming normal distributions, and therefore, the criticism goes, the ranking logic must be inaccurate. 

In a recent Medium article, Giora Omer offers an alternative min-max normalization approach, which would preserve the relative distribution of each category, without over- or under-valuing the outliers. But while the general approach is discussed, rankings are neither demonstrated nor compared. Josh Lloyd at Basketball Monster recently promised an improvement using his DURANT method (Dynamic Unbiased Results Applying Normalized Transformations), which, although inspiring comments like, “Josh Lloyd is trying to crack the fantasy Da Vinci code!”, remains opaque. Lloyd points out that percentage stats seem to vary differently than other, 'counting stats,' but does not offer a reasons nor a method to approach it any differently. A Reddit user recently offered a so-called CARUSO method, which promises to use ML for custom category transformations, but how so remains a mystery, and as some readers note, the method does not seem to deliver. The only ranking method that I have seen fully described is Zach Rosenof’s *G*-Score method, which improves on *Z*-Score rankings by accounting for period-level (i.e., week-to-week) variance in category accumulation; because some categories are more volatile, investing in them is risky. Rosenof’s is also the only paper to compare different metrics, finding empirical support for his *G*-Score rankings method in simulated head-to-head matchups against hypothetical teams using *Z*-Score rankings.

### Actions

Here, I develop and describe six different ranking algorithms of my own, and compare them head-to-head against ESPN, Yahoo.com, and Basketball Monster. Each of my ranking methods applies a Sigmoid Harmonic Weight to shot-Attempts for percentage category transformations (hence SHAW: Sigmoid-Harmonic Attempts-Weighted - transformations). This approach aims to reduce distortion from outliers and enhance the signal from players contributing efficiently across categories. The lambda parameter in the sigmoid function is dynamically tied to the skew of attempts, while the sensitivity is a function of attempt variance, thus creating a context-sensitive weighting mechanism. From there, my algorithms follow some familiar and some novel ideas. 

| **Ranking Algorithm** | **Description** | **Strengths** | **Weaknesses** |
|-----------------------|-----------------|---------------|----------------|
| SHAW-Z                | Sum of standardized counting stats and sigmoid weighted percentages | Preserves category dispersion while setting category distributions to like terms | May over-value outliers |
| SHAW-mm               | Sum of Min-Max (0-1) scaled counting stats and sigmoid weighted percentages | Preserves category dispersion without overvaluing outliers | May under-value real differences |
| SHAW-Scarce-mm        | SHAW-mm ranking with weights applied to scarce categories | Rewards scarce category producers | Scarce categories might also vary unstably from week to week |
| SHAW-rank-sum         | Reverse-ranked sum of each-category ranks (using sigmoid weighted percentages) | Simple math; preserves relative player *position* in each category | Does not account for category variance |
| SHAW-H2H-each         | Ranked sum of individual-player categories won vs. the field (using sigmoid weighted percentages) | Rewards balance across categories | Does not account for category variance |
| SHAW-H2H-most         | Ranked sum of individual-player matchups won vs. the field (using sigmoid weighted percentages) | Rewards consistency in top categories | Does not account for category variance |

Rankings are then compared using two methods: 1. A top-*n* players 'super-team' for each ranking, comparing teams head-to-head across the nine, standard categories; 2. A head-to-head super-tournament of 10 simulated (snake-drafted) teams from each metric, counting the category and matchup wins for each team and for each metric that the team came from. 

Along the way, I build an ETL (Extract, Transform, Load) pipeline that starts with up-to-date NBA player stats and ends with an interactive player-ranker dashboard. 

### Results

All six SHAW-transformation algorithms beat ESPN’s rankings and three of the six beat Yahoo’s rankings when comparing the top 100 players from each. In a separate test, two of the six beat Basketball Monster when comparing the top 100 players. These results are robust at different depths - i.e., top 20, top 50, top 130. Surprisingly, the top performing algorithm was the SHAW-Z ranking, which is most like the oft-criticized method that the major platforms use. The fact that mine came out on top attests to the novel treatment of percentages. The second most successful algorithm was the SHAW-Scarce-mm ranking, a model based on min-max scaling that adds a modest bonus for players that outperform in scarce categories. Although I don’t compare my rankings to Rosenof’s G-Score ranking method, I note that we come to opposite conclusions regarding the value of high performers in scarce, volatile categories. 

### Growth/Next Steps

As a research and data science project, I could not be happier. My rankings beat the competition. I hope that by publishing the actual ranking methods AND their systematic comparisons that this project inspires further research and discussion. I also leave plenty of room for improvement and building in terms of metric construction, ranking comparison, and front-end dashboard development.

# Extraction

I use the [NBA API](https://github.com/swar/nba_api/blob/master/README.md?utm_source=chatgpt.com). An unofficial but widely used source for up-to-date NBA statistics.

```python
from nba_api.stats import endpoints
from nba_api.stats.endpoints import LeagueDashPlayerStats
import pandas as pd
import os 

os.chdir("C:/...Projects/NBA")

season = "2024-25"

def fetch_player_stats():
    player_stats = LeagueDashPlayerStats(season=season)
    df = player_stats.get_data_frames()[0]  # Convert API response to DataFrame
    return df

nba_24_25 = fetch_player_stats()

nba_24_25.to_csv("data/nba_2024_25.csv", index=False)

```

The code above returns a dataframe of 66 statistical columns and 550+ rows (players), which get added every time a new players sees the court in the season. 

A select sample of the raw data, showing three superstars and one lesser-known player. 

| **Player Name** | **GP** | **FGA** | **FGM** | **REB** | **AST** | **TOV** | **STL** | **BLK** | **PTS** |
|-----------------|--------|---------|---------|---------|---------|---------|---------|---------|---------|
| LeBron James | 62 | 1139 | 583 | 506 | 525 | 238 | 58 | 35 | 1521 |
| Nikola Jokic | 63 | 1226 | 706 | 806 | 647 | 206 | 109 | 43 | 1845 |
| Victor Wembanyama | 46 | 857 | 408 | 506 | 168 | 149 | 52 | 176 | 1116 |
| Jordan Goodwin | 20 | 104 | 50 | 77 | 29 | 16 | 23 | 11 | 130 |


# Transformations

In a typical (*category*) fantasy basketball league, a team’s weekly score is the sum of players’ cumulative stats in each of the following 9 categories: 
- Field Goal Percentage (FG_PCT = FGM/FGA)
- Free Throw Percentage (FT_PCT = FTM/FTA)
- Points (PTS)
- 3-point Field Goals made (FG3M)
- Rebounds (REB)
- Assists (AST)
- Steals (STL)
- Blocks (BLK)
- Turnovers (TOV) (which count negatively) 

From there, category leagues can keep score one of two ways: a weekly ‘head-to-head’ *matchup* win against another team (for the team that wins the most categories), or a win for *each* category (with all nine categories at stake). Leagues can be customized to include more or less categories, but the default is nine, and the main platforms’ rankings are based on these categories, so that is what I focus on in this write-up. Separately, I also go on to construct rankings based on my own 11-category leauge, which includes 3-Point Feild Goal Percent and Double-Doubles. I'm interested in the following columns: 

```python

nba_subset = nba_24_25[['PLAYER_NAME', 'GP', 'FGA', 'FGM', 'FTA', 'FTM','FG3A', 'FG3M', 
             'REB', 'AST', 'TOV', 'STL', 'BLK', 'PTS', 'DD2', 
             'FG_PCT', 'FG3_PCT', 'FT_PCT']].copy()

```

The goal is to turn raw NBA statistics from the relevant categories into a single player ranking metric that will help team managers understand a single player’s relative fantasy value. It would be easy if there was one statistical category to keep track of – players would be ranked by the number of points they get, for example. (In fact, *Points* leagues are an alternative scoring format that assigns every statistical category a uniform ‘point’ value, which is too easy to be any fun.) But category leagues require comparisons across 9 different measures. How can you value a player that gets a lot of points and few rebounds vs. one that gets a lot of rebounds and few assists? How do you compare a player that shoots an average free throw percentage on high volume, to a player that shoots above average on low volume? 

Before we get into that, it is important to understand the distributions. 

### Distributions

Most NBA stat categories are positively skewed. Few players get the most points, rebounds, assists, etc. 

![alt text](/img/posts/PTS_REB_raw.png "Raw Distributions, Points & Rebounds")

At the same time, a large number of players accumulate very little, if any, stats at all. THIS fact is important, although it is hardly considered among the experts. When we start to transform distributions, there is a huge difference in comparing the fantasy-relevant players vs. the entirety of the league. That is, the relative value of a player for fantasy purposes should be considered only in relation to other plausible options, and NOT in relation to the bulk of players that get little or no playing time. 

Before making any transformations, thus, a cut-off should be levied, and I define mine at the 25th percentile of games played. Granted, this number is arbitrary, but it is a useful starting point that has the effect of eliminating most ‘garbage time’ players as well as those with season defining injuries (e.g., Embiid in 24-25). Then, I scale raw category accumulations by games played. Most fantasy relevant players will miss some time during a season due to minor injuries, so using per-game statistics (which is standard) helps to level the field. 

```python

# Set Games Played threshold for counting stats
minimum_games_threshold_quantile = 0.25
min_games = stats["GP"].quantile(minimum_games_threshold_quantile)

pg_stats = stats[stats["GP"] >= min_games].copy()

# reverse code turnovers
pg_stats['tov'] = pg_stats['TOV'].max() - pg_stats['TOV']

raw_categories = ['FGA', 'FGM', 'FTA', 'FTM', 'FG3M', 'PTS',
             'REB', 'AST', 'STL', 'BLK', 'tov']

for cat in raw_categories:
    pg_stats[cat + "_pg"] = pg_stats[cat] / pg_stats['GP'] 

```

The result is a more modest 400 + player pool, with the large number of zeros eliminated. 

![alt text](/img/posts/PTS_REB_PG.png "Per-Game Points & Rebounds")


## Percentages

Percentage categories deserve special treatment because they are a function of two distributions: makes and attempts. We don’t simply add percentages like we do the other categories: we divide total team makes by total team attempts to get a final percentage score for each weekly matchup. 

![alt text](/img/posts/FT_PCT_vs_A.png "Free Throw Distributions")

A player that shoots an average percentage on high volume of attempts, has a larger impact on a fantasy matchup than an above average shooter that rarely shoots. To evaluate a player’s value in a percentage category, thus, shot volume needs to be considered alongside percent made. The recieved method for doing this is to measure a player's *impact* by finding the difference between their number of makes and what is expected given their number of attempts and league percentage averages. Standardizing the impact thus gives a comparable value score for that category. For example: 

```python

league_pct = pg_stats["FTM"].sum() / pg_stats["FTA"].sum()
FT_impact = pg_stats["FTM"] - (pg_stats["FTA"] * league_pct)

```

At first glance this would seem fair, a player’s percentage impacts a team’s total to the extent that they take above or below average shot attempts. But because attempts are positively skewed, and percentages are negatively skewed, this method can produce some extreme numbers in both tails. Standardizing yeilds the following Z-Score distribution:

![alt text](/img/posts/FT_Impact_Z.png "Free Throw Distributions")

The test case here is Giannis Antetokounmpo. He shoots a sub-par 60% (the league average is 78.2%), but he is also a league-leader in attempts (as an elite scorer otherwise, he gets fouled a lot). Giannis's Free Throw Z-Score is -8. 


| **Player Name** | **FT%** | **FTM** | **FTA** | **FT Impact Z-Score** |
|-----------------|---------|---------|---------|-----------------------|
| Giannis Antetokounmpo | 60.2 | 369 | 613 | -8.12 | 
| Steph Curry | 92.9 | 252 | 234 | 2.73 | 
| Shae Gilgeous-Alexander | 90.1 | 563 | 625 | 5.50 | 
| James Harden | 87.4 | 515 | 450 | 3.51 | 
| LeBron James | 76.8 | 289 | 222 | -0.23 | 


A critical question for fantasy category valuations is thus, does Giannis hurt you THAT much? Does Shae Gilgeous Alexander *help* you that much? 

For other, counting statistics, I will argue later that skewed distributions are meaningful and useful, but a percentage statistic is bound between 0 and 1, so positive and negative constributions to it are limited: they are asymptotic, not linear! 

[IMAGE HYPOTHETICAL distribution of attempts in relation to percentage] 

## SHAW Percentage Transformation

It seems reasonable, then, that percentage categories should be treated with this observation in mind. Attempts should be weighted within limits (defined by its distribution) to reflect the non-linearity of their effect on percentages. 

I apply a Sigmoid-transformation to attempts to get a weight value. The **Sigmoid transformation** is defined as follows:

$$
\text{Weight} = \left( \frac{S - 1}{0.5} \right) \left( \frac{1}{1 + e^{-k (x - 1)}} - 0.5 \right) + 1
$$

Where:

- x = Attempts / Average Attempts
- S = 1 + CoV (Coefficient of Variation)
- k = 1 / (1 + |Skewness|)

Applying to attempts in a percentage category, this yields weight values of 1 when a player's attempts are at the league average, and a maximum of 1 + CoV (which for Free Throws is 2.17). For attempts below average, the player is assigned a weight value below one, which theoretically approaches zero for players with zero attempts.

[IMAGE: My sigmoid curve vs. linear attempts/average] 

I then apply the weight directly to the percentage *deficit*, or difference from the mean, which I cap at 3 standard deviations from the mean, effectively limiting the impact that a few terrible shooters have on the rest of the distribution. 

```python
# Define Sigmoid Weight for Attempts
def sensitivity(attempts): 
    return 1 + attempts.std() / attempts.mean() # 1 + CoV for maximum weight

def lambda_val(attempts): 
    return 1 / (1 + abs(skew(attempts))) # 1 / 1 + skewness for modest slope

def sig_weight(attempts):
    
    avg_attempts = attempts.mean()
    # Ratio of attempts to average
    x = attempts / avg_attempts
    
    k = lambda_val(attempts)
    S = sensitivity(attempts)

    # Calculate the sigmoid transformation
    sigmoid = 1 / (1 + np.exp(-k * (x - 1)))
    
    # Scale and center the sigmoid function
    weight = ((S - 1) / 0.5) * (sigmoid - 0.5) + 1
    
    return weight

# Define function to transform a percentage stat
def SHAW_transform(nba, stat, attempts, makes):

    # League Average Percentage (using eligible (25+ GP) players)
    league_avg = nba[makes].sum() / nba[attempts].sum()

    # Deficit = percentage - average
    nba[f"{stat}_deficit"] = -(global_avg - nba[stat])
    
    # Clip extreme deficits to prevent outliers from dominating
    max_deficit = nba[f"{stat}_deficit"].std() * 3
    nba[f"{stat}_deficit"] = nba[f"{stat}_deficit"].clip(lower=-max_deficit, upper=max_deficit)
    
    # Weight = sigmoid transformed attempts 
    nba[f"{stat}_weight"] = sig_weight(nba[attempts])

    # SHAW transformation = Deficit * Weight
    nba[f"SHAW_{stat}"] = nba[f"{stat}_deficit"] * nba[f"{stat}_weight"]

    # Finally, remove players not meeting minimum attempts threshold from the distribution (while keeping the rows)
    min_attempts = nba[attempts].quantile(minimum_attempts_quantile)
    nba.loc[stats[attempts] < min_attempts, f"SHAW_{stat}"] = np.nan

    return stats

# Apply transformation function to FT%, FG%, and 3P%
nba = SHAW_transform(nba, "FT_PCT", "FTA", "FTM")
nba = SHAW_transform(nba, "FG_PCT", "FGA", "FGM")
nba = SHAW_transform(nba, "FG3_PCT", "FG3A", "FG3M")

# so that 0 percents don't become nans
SHAW_percentages = ['SHAW_FT_PCT', 'SHAW_FG_PCT', 'SHAW_FG3_PCT']
for col in SHAW_percentages:
    nba[col] = nba[col].fillna(0)

```

The resulting distribution is the sigmoid-harmonic attempts-weighted deficit, which could be added back to a player's raw percentage, or left alone, because the resulting dispersion will be the same. 

![alt text](/img/posts/SHAW_FT_deficits.png "Free Throw Distributions")

SHAW-transformed percentages thus follow a reasonably normal distribution that can be appropriately scaled to compare to other cumulative categories. If standardizing, for example, note that the extremes that are produced in the tails (Giannis vs. Shai G-A for example) using Impact scores are muted by SHAW-tranformations. This method thus undervalues SGA and overvalues Giannis relative to existing rankings. It ammounts to a -2.2 point swing for SGA and + 4.6 point swing for Giannis in *Z*-score ranking systems, which is plenty enough to change their positions (except for the fact that SGA is a top 5 player anyway, he hardly moves). 


| **Player Name** | **FT %** | **FTM** | **FTA** | **X:atts/avg** | **FT Impact Z_Score** | **Deficit** | **Sig-weight** | **SHAW %** | **SHAW-Z-Score** | 
|-----------------|---------|---------|---------|-----------------|-----------------------|-------------|----------------|---------------------|------------------|
| Giannis Antetokounmpo | 60.2 | 369 | 613 | 5.59 | -8.12 | -0.18 | 1.65 | 48.6 | -3.55 | 
| Steph Curry | 92.9 | 252 | 234 | 2.30 | 2.73 | 0.15 | 1.21 | 96.1 | 2.30 | 
| Shae Gilgeous-Alexander | 90.1 | 563 | 625 | 5.70 | 5.50 | 0.12 | 1.66 | 98.0 | 2.52 | 
| James Harden | 87.4 | 515 | 450 | 5.70 | 3.51 | 0.09 | 1.56 | 92.5 | 1.85 |  
| LeBron James | 76.8 | 289 | 222 | 2.63 | -0.23 | -0.01 | 1.27 | 76.4 | -0.13 | 


## Standardization

Standardization is the dominant method because is works. It transforms unique distributions to the same theoretical range so that they can be easily compared. A player’s standardized-, or *Z*-score is their relative difference (above or below average), divided by the standard deviation of the distribution.

In Python: 

```python

def Z(stat):
    return (stat - stat.mean())/stat.std()

```

I create a SHAW-Z ranking by including my SHAW-tranformed percentages in the set of categories to be standardized, taking the sum of those Z-Scores, and returning a rank-order. 

```python

nine_cat = ['SHAW_FT_PCT', 'SHAW_FG_PCT', 'PTS', 'FG3M', 'REB', 'AST', 'STL', 'BLK', 'tov']

def Z_rank(df, categories, metric_label):
    df[metric_label] = Z(df[categories]).sum(axis=1) # rank and add the categories 
    df[metric_label + "_rank"] = df[metric_label].rank(ascending=False, method='min').reindex(df.index) # return rank order
    return df

pg_stats = Z_rank(pg_stats, nine_cat, 'SHAW-Z')

```


## Min-Max Scaling

Similar to standardization, min-max scaling (transforming the range to 0 – 1) preserves the spread, and it also limits the range, so outlier values are reigned in. This means that if a player is an outlier in a category, their maximum value in a category is limited at 1, and all other players are scaled accordingly. This makes intuitive sense, because each category is only worth 1 point. If a fantasy team over-performs in a category, they don't get more than one point. 

To scale between 0-1, take the difference from the minimum value and divide by the range: 

x - min(X) / (max(X) - min(X))

Because this approach is standard for many machine learning tasks, the conventional method in Python makes use of the ML SciKitLearn library. 

```python

from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler() # default range = 0-1
# scaler.fit_tranform(x)

```

I create a SHAW-mm ranking by including my SHAW-tranformed percentages in the set of categories to be scaled between 0 and 1, taking the sum of those Min-Max scores, and returning a rank-order. 

```python

def minmax_rank(df, categories, metric_label):
    scaler = MinMaxScaler() # default range = 0-1
    df[metric_label] = scaler.fit_transform(df[categories]).sum(axis=1) # rank and add the categories 
    df[metric_label + "_rank"] = df[metric_label].rank(ascending=False, method='min').reindex(df.index) # return rank order
    return df

pg_stats = minmax_rank(pg_stats, nine_cat, 'SHAW-mm')

```

## Scarcity Ranking

Scarcity is the basis of modern economics because value is a function of scarcity (c.f., David Ricardo, 1817). In an NBA game, there are plenty of points scored and many players accumulate points. By contrast, blocked shots or steals might happen a handful of times in a game, and only a few players across the league tend to excel in those categories. 

Although Blocks and Points count the same in terms of categories, having a player that excels in Blocks may be more valuable than a high Points getter, because the Block star is harder to replace. There are fewer elite shot blockers in the league, and if your team doesn't have one, you may have a hard time competing in that category. 

To test this hypothesis, I developed an index that weighs the relative scarcity of each of the seven cumulative categories (on a scale of 0-1, total scarcity = 1) by subtracting the skew from the inner-quartile range, and normalizing (min-max scaling) the results. Then, for the min-max transformed categories, I multiply the normalized category distributions by its scarcity score. Because both the scarcity index and normalized distributions range between 0-1, the resulting sum of scarcity weighted scores also range between 0 and 1. The result is a modest addition that boosts a player’s min-max score by a maximum of one point. That should be enough to redistribute the players to test whether rewarding scarcity actually improves the rankings. 

``` python

def scarcity_rank(df, categories, metric_label):
    
    # Calculate skewness and interquartile range for each category
    skewness = df[categories].apply(skew)
    iqr = df[categories].quantile(0.75) - df[categories].quantile(0.25)

    # Compute scarcity index (skewness / IQR) and scale it to [0, 1]
    scarcity_index = (skewness / iqr).sort_values(ascending=False)
    scarcity_weights = (scarcity_index - scarcity_index.min()) / (scarcity_index.max() - scarcity_index.min())

    scaler = MinMaxScaler() 
    # Scale each category and apply scarcity weights
    for stat in categories:
        # Scale the category to a 0-1 range
        df[f'{stat}_mm'] = scaler.fit_transform(df[[stat]])
        
        # Apply the scarcity weight to the scaled category
        df[f'{stat}_mm_scarcity_weight'] = df[f'{stat}_mm'] * scarcity_weights[stat]

    # Sum the scarcity-weighted stats
    df[f'{metric_label}_scarcity_boost'] = df[[f'{stat}_mm_scarcity_weight' for stat in categories]].sum(axis=1)
    
    # Add the boost to the original min-max scaled metric
    df[f'{metric_label}_mm_total'] = df[[f'{stat}_mm' for stat in categories]].sum(axis=1)
    df[f'{metric_label}_scarcity_score'] = df[f'{metric_label}_mm_total'] + df[f'{metric_label}_scarcity_boost']

    # Rank based on the final scarcity score
    df[f'{metric_label}_scarcity_rank'] = df[f'{metric_label}_scarcity_score'].rank(ascending=False, method='min')

    return df

counting_stats = ['PTS', 'FG3M', 'REB', 'AST', 'STL', 'BLK', 'tov']

pg_stats = scarcity_rank(pg_stats, counting_stats, '9_cat')

```

## Other Ranking Methods

Other plausible ranking methods use the SHAW transformed percentages but forgo additional transformations and concerns with category variance. 

### Ranked Sum of Category Ranks

The simplest method is to rank players in each category, add up those ranks, and reverse rank that sum. 

```python

def cat_rank_sum(df, categories, metric_label):
    category_ranks = pd.DataFrame()
    for stat in categories: 
        cat_ranks[stat + "_rank"] = df[stat].rank(ascending=False, method='min')
    df[metric_label] = cat_ranks.sum(axis=1)
    df[metric_label + "_rank"] = df[metric_label].rank(ascending=True, method='min')
    return df

pg_stats = cat_rank_sum(pg_stats, nine_cat, 'SHAW_rank_sum')

```

This method does not preserve the relative spread, but instead distributes players uniformly in each category, while still accounting for the relative position of each player in each category. 
And since we’re comparing all the players across all categories, this method seems elegant. But results are somewhat surprising, as we'll see.

### Head to Head individual player comparisons

Another approach to ranking involves observing how players match up head-to-head (H2H) against other players. After SHAW-transforming the percentages, players are compared against every other player in every category. This requires building a data frame with a row for each player combination. From there we can count the number of categories that each player wins versus each other, we can assign a matchup winner (for most categories won), and we can count the total categories, and total matchup wins against the field. 

This takes considerable computing time, so it’s not a practical method to build into an ETL pipeline. For the sake of comparison, I do construct the rankings based on H2H matchup wins and H2H each category wins, and compare these to the other metrics. 

For brevity, the code for this can be found in my GitHub repository. 

The six different ranking methods produce a lot of similar rankings, but enough variation to be meaningfully different, and which can be compared to ESPN, Yahoo, and Basketball Monster.

| **Player Name** | **Traditional Z-rank** | **SHAW Z rank** | **SHAW mm rank** | **SHAW Scarce mm rank** | **SHAW rank-sum rank** | **SHAW H2H each rank** | **SHAW H2H most rank** | **ESPN** | **Yahoo** | **Basketball Monster** |
|---|---|---|---|---|---|---|---|---|---|---|
| Nikola Jokic | 1 | 2 | 1 | 2 | 2 | 2 | 1 | 5 | 1 | 1 |
| Shai Gilgeous-Alexander | 1 | 3 | 3 | 3 | 1 | 1 | 2 | 1 | 3 | 2 |
| Victor Wembanyama | 3 | 1 | 2 | 1 | 3 | 3 | 3 | 111 | 2 | 3 |
| Karl Anthony-Towns | 5 | 7 | 7 | 8 | 4 | 4 | 4 | 6 | 4 | 6 |
| Steph Curry | 10 | 10 | 8 | 9 | 21 | 20 | 34 | 16 | 12 | 13 |
| James Harden | 16 | 17 | 17 | 15 | 14 | 15 | 10 | 12 | 27 | 19 |
| LeBron James | 21 | 23 | 22 | 19 | 19 | 19 | 13 | 17 | 14 | 21 |
| Giannis Antetokounmpo | 81 | 26 | 43 | 28 | 90 | 89 | 75 | 3 | 23 | 76 |
| Dyson Daniels | 31 | 30 | 72 | 47 | 92 | 92 | 76 | 51 | 31 | 15 | 


## The Competition

I mentioned earlier that there is no transparency as to how the major platforms compute rankings, and to be fair, Yahoo and ESPN’s fantasy player portals don’t actually include rank numbers after the season begins. Nevertheless, they do provide an ostensibly rank-ordered list that can be sorted based on season totals, per-game averages, two week averages, etc. Because the order changes based on the parameter, and because the top players are congruent with other rankings, it is reasonable to assume there is a ranking algorithm working behind the scenes. 

I copy/pasted the top 200 players for season averages (a.k.a. per-game) statistics on March 26, 2025 from both ESPN and Yahoo. I mention the date because this is a single point-in-time comparison. I refreshed my own rankings on March 26, so that I’m comparing player rankings using the same player statistics.

ESPN provides a separate ‘updated category rankings’ list every week, but these are based on future projections – designed to help fantasy managers make replacement decisions  -and they are different from the lists provided in their fantasy player portal. Still, it does appear that ESPN uses some type of *forecast* ranking system, that accounts for injuries, even for their “season averages” list. Why Victor Wembanyama was listed by ESPN at the 111th position on their list, however, is beyond me. He should either be a top 10 player by season averages, or he should be completely removed (for forecasting purposes) because he suffered a season-ending blood condition mid-way through the seaon

Nevertheless, to make the comparisons fair, I removed all the currently injured players in ESPN’s top 130 from the pool of eligible players. That injury list includes: 

[INJURY LIST]

With those indiscrepancies out of the way, there were also a few players in Yahoo’s and ESPN’s average rankings that were filtered out of my own player pool because they did not meet my minimum games threshold. These include Joel Embiid, who has only played X games this season, and [X] a player whose playing time increased only towards the end of the season. Although these players had ranks elsewhere, they were not in the eligible player pool and thus were removed from rank comparisons. 

I compared Basketball Monster rankings separately. Due to a mistake on my part, I did not scrape their rankings until 3/28/2025. Since two more days of NBA games had passed, I decided to compare these separately, and refreshed my own rankings on 3/28/2025 to compare them. Basketball Monster keeps a very clear ranking system that appears to be true to real “season averages”. I did not have to exclude any injured players, but players that did not meet the minimum games threshold were removed by default. 

## Top N Players
First, I simulate head-to-head, 9-category matchups using the top n players in each ranking system. I compare the top 20, top 50, top 100, and top 130 (the number of active players in a 10-team league) in separate matchups using real, up-to-date, per-game statistics. 

[PYTHON CODE CHUNK]

There is of course overlap between the metrics, but there is enough difference to put the algorithms to the test. For example, a punishing -6 (standard attempt weighted) Z-score in the free throw percentage category would put Giannis out of the top 20, while a modest -3 (SHAW weighted) Z-score might ensure that he remains. If Giannis were excluded from the upper lists, he would likely be reincluded by the time we reach top 100 and top 130 players. As n gets larger, however, we should expect there to be more variation between ranking systems.

IMAGE Results table. 

The results of this experiment seem clear, if not conclusive. The top performing ranking system is…. X, 
Yahoo, and Basketball Monster also offer competitive rankings. ESPN’s rankings don’t match up well, and this is likely due to the confusion (and lack of transparency) about what ESPN’s rankings actually mean. They appear to give extraordinary weight to a player’s short-term statistics, which is GREAT if you are looking for a replacement player at any point in time. Still, when toggling ‘all players’ and ‘season averages’ this is the ordered list that they provide. 

## Snake-Draft Tournament
A second method of comparing rankings is to simulate 10 teams from each, which should provide an approximately average team from that ranking metric. But the results of head-to-head matchups among the scores of teams using this method offer a drastically different picture. 

[IMAGE: Top and bottom 10 teams]. 
Given the results of the top-n teams, we might expect that the SHAW-Z and SHAW-Scarcity-Boosted-mm ranked teams would rise to the top. Indeed these metrics take the top to matchup-wins spots, but the Basketball Monster rankings get the most total category wins. Perhaps more interestingly, the differences among the top performers appear insignificant. Shaw-Z wins only 1% more than the next. 





Unpacking the team team-level standings further, we see that same ranking systems perform across the range of wins and losses. In fact, the H2H rankings are among the top winners (and losers) when they did not perform well at all in top-n teams matchups. This experiment leads me to think that – all other things being equal – the luck of the draw – in this case the draft order given slightly different combinations of players – matters the more than rankings themselves. 

## Ranking Matchup Summary:
I am proud to say that many of my rankings outperformed the competition in head-to-head matchups. Specifically, if my Z-ranking beat other rankings that are based on standardization (specifically Yahoo and BBM), the difference can be found in the novel way that I treat percentage transformations. 
Additionally, the relative success of my ‘scarcity boosted’, min-max ranking should demonstrate that scarcity does matter, and this finding should inform future improvements. 
It may also be worth noting the limitations of my experiments. The simulated tournament revealed the surprising finding that, when dividing any rank-based player pool into n teams, the variances that made the whole cohere become unstable, and draft order and team build might weigh more heavily. In fact, my snake-draft simulation did not include player position rules, which are an integral part of fantasy team building. It is likely that, for each ranking metric, the snake draft produced a set of teams where certain positions (and thus certain statistical contributions) were over or underrepresented. 

## Value over Replacement
I construct one final metric for my player-ranker dashboard, a VORP score – which I construct in two forms – and which is simply a player’s SHAW-Z score or SHAW-mm score in relation to the average of the 100-130 ranked players. 
Because Z-scores rankings reward high achievers in specific categories, and because Min-Max scores reflect balance across categories (i.e., no one category over shadows another), I rebrand these “Impact Value” and “Balance Value”, respectively, and give the user a change to evaluate players in those terms for their unique team needs. 

[PYTHON CODE CHUNK]

# Load / Endpoint / Streamlit App
Finally, as a proof of concept, I build a Streamlit App as a user endpoint. The app allows the user to select among my top ranking metrics (except the H2H ones that take computing time). 
It is not yet deployed outside of my local drive, but my near-future goal is to make this publically accessible, executing the whole ETL pipeline described above. 


# Conclusion/Growth

think it is safe to conclude that rankings only make up a small part fantasy success. Even when using the most competitive method (z scores), there is still intransitivity among players, and then the draft order generates noise that rankings may not be able to overcome. 
Even more confounding would be the actual position requirements in fantasy leagues, which my simulated matchups do not account for. Savvy fantasy managers know as well that building a team is equally art and science. Knowing that you only need 5 of 9 categories to win, for example, you can elect to ‘punt’ or exclude one or more categories from your ideal team build. Knowing which categories to punt depends on what players are available at what order during draft day. A ranking system that accounted for every punt scenario in real time during a player draft would be overwhelming to say the least. 
And finally, I have not addressed factor that all fantasy managers know to be the one the real difference maker basketball – injuries. This year, perennial MVP favorite Nikola Jokic missed 5 straight games during fantasy playoff weeks. If you had him on your team, you were likely in the playoffs, and then you likely lost because he was injured. 

### PERCENTAGES DO NOT SCALE LINEARLY

### COMMENTS ON STANDARDIZATION and its CRITICS

The criticism that Z-scores over-value outliers in skewed distributions turns out to be wrong. My standardized rankings beat all the other rankings, but WHY? It turns out that standardization, like my scarcity boost formula, rewards scarcity. When a distribution is highly skewed, that may be because the event is scarce, or because elite producers in that category are scarce. In fact, eliteness produces skew, and thus scarcity. Victor Wembanyama pushes out the tail of the distribution because he is a unique talent; the skew results from the fact that there is no one else out there with him. Wemby’s Z-score of 8 in Blocks, or Dyson Daniel’s Z-score of 6 in Steals not only reflect those player’s ability in relation to those categories, but they reflect their scarcity in relation to other players as well. 

To say that standardization assumes a normal distribution, may be like saying wearing running shoes assumes that I am running. The misconception comes from the statistical concept of the central limit theorem, which allows us to assign probabilities to independent events occurring because of the fact that sample distributions ARE normally distributed when numbers are large. Then, standard deviations have real probabilities attatched. But we know that NBA stats aren’t normal, and we are not trying to test whether or not Wemby’s 3.5 blocks per game comes from the same distribution as LeBron Jame’s [X] Blocks. We already know they do. I am however, interested in knowing how Wembanyama’s 3.5 blocks per game compares to Sabonis’ 14 rebounds, or Trae Young’s 10+ assists. Standardization simply gives us a like metric to compare categories that otherwise have radically different ranges. The fact that it produces some astounding numbers at the tails is actually a benefit for comparing cumulative fantasy categories when the few players that can achieve those results can only be picked once. 







