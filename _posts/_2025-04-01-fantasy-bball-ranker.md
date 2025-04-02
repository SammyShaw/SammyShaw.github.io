---
layout: post
title: Fantasy BBall Ranking using SHAW-Transformation
image: "/posts/PLACEHOLDER.png"
tags: [ETL Pipeline, Statistics, System Ranking, Python, Fantasy Basketball]
---

In this project I create six unique fantasy basketball ranking algorithms and compare them to the current leading fantasy platform ranks: ESPN, Yahoo.com, and Basketball Monster. Each of my rankings uses a Sigmoid-Hierarchical Attempts-Weighted (SHAW) percentage transformation. Comparing rankings Head-to-head using the top 100 players from each, two of my six ranking systems beat Basketball Monster, three beat Yahoo, and all six beat ESPN. I extract, transform, and load data for my own fantasy basketball player ranking app. [URL]. 

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

Fantasy sports are a popular past time with over 62.5 million participants in the U.S. and Canada [FSGA.org]. In a fantasy sports league, participants (or, team managers) create a team by competitively selecting from a pool of professional players, whose real stats become their own fantasy stats for the season. Choosing a successful team requires knowing the players’ and understanding the league’s scoring format. In 9-category fantasy basketball leagues (the standard competitive fantasy basketball format), teams are compared – and thus players must be evaluated - across nine different measures. To make this easier, the major platforms Yahoo.com and ESPN include player rankings to help managers make their choices. But Yahoo and ESPN often include questionable players in their top ranks, and unfortunately, neither platform is transparent about how their rankings are calculated. As a fantasy basketballer, I’ve often thought that these rankings could be improved. 

### Literature Review

I’m not the only one to question the platform rankings and there are some interesting contributions out there concerning the best way to construct them. The consensus in these discussions is that the existing (Yahoo and ESPN) rankings are made by standardizing the categories and then ranking the sum of the players’ Z-scores (with caveats for percentage categories). But standardization is then criticized for over- or under-valuing outliers if distributions are skewed (which they are in the NBA). Victor Wembanyama’s Z-score of 8 in the Blocks category, for example, statistically represents a nearly impossible occurrence (assuming normal distributions), and therefore, the criticism goes, the ranking logic must be inaccurate. 

In a recent Medium article, Giora Omer offers an alternative min-max normalization approach, which would preserve the relative distribution of each category, without over- or under-valuing the outliers. But while the general approach is discussed, rankings are neither demonstrated nor compared. Josh Lloyd at Basketball Monster recently promised an improvement using his DURANT method (Dynamic Unbiased Results Applying Normalized Transformations), which, although inspiring comments like, “Josh Lloyd is trying to crack the fantasy Da Vinci code!”, remains opaque. A Reddit user recently offered to expound on this with a so-called CARUSO method, which promises to use ML for custom category transformations, but how so remains a mystery, and as some readers note, the method does not seem to deliver. The only ranking method that I have seen fully described is Zach Rosenof’s G-Score method, which promises to improve on Z-Score rankings by accounting for period-level (i.e., week-to-week) variance in category accumulation. Rosenof’s is also the only paper to compare the differences, finding empirical support for his G-Score rankings method in simulated head-to-head matchups against hypothetical teams using Z-Score rankings.

### Actions
Here, I develop and describe six different ranking algorithms of my own, and compare them head-to-head against ESPN, Yahoo.com, and Basketball Monster. 
Each of my ranking methods applies a Sigmoid Transformation to shot-attempts, weighting deficits for percentage categories based on their underlying distribution (hence SHAW – Sigmoid-Hierarchical Attempts-Weighted - transformations). This approach aims to reduce distortion from outliers and enhance the signal from players contributing efficiently across categories. The lambda parameter in the sigmoid function is dynamically tied to the skew of attempts, creating a context-sensitive weighting mechanism. From there, my algorithms follow some familiar and some novel ideas. 

| **Ranking Algorithm** | **Description** | **Strengths** | **Weaknesses** |
|-----------------------|-----------------|---------------|----------------|
| SHAW-Z                | Sum of standardized counting stats and sigmoid weighted percentages | Preserves spread while setting category distributions to like terms | May over-value outliers |
| SHAW-mm               | Sum of Min-Max (0-1) scaled counting stats and sigmoid weighted percentages | Preserves spread and normalizes without overvaluing outliers | May under-value real differences |
| SHAW-Scarce-mm        | SHAW-mm ranking with weights applied to scarce categories | Rewards scarce category producers | Scarce categories might also vary unstably from week to week |
| SHAW-rank-sum         | Reverse-ranked sum of each-category ranks (using sigmoid weighted percentages) | Simple math; preserves relative player *position* in each category | Treats categories uniformly without accounting for variance |
| SHAW-H2H-each         | Ranked sum of total categories won vs. the field (using sigmoid weighted percentages) | Rewards balance across categories | Treats categories uniformly without accounting for variance |
| SHAW-H2H-most         | Ranked sum of total matchups won vs. the field (using sigmoid weighted percentages) | Rewards consistency in top categories | Treats categories uniformly without accounting for variance |


Rankings are then compared using two methods: 1. A top-n players list for each ranking, comparing players head-to-head across the nine, standard, categories; 2. A head-to-head super-tournament of 10 simulated (snake-drafted) teams from each metric, counting the category and matchup wins for each team and for each metric that the team came from. 
Along the way, I build an ETL (Extract, Transform, Load) pipeline that starts with up-to-date NBA player stats and ends with an interactive player-ranker dashboard. 

### Results

All six SHAW-transformation algorithms beat ESPN’s rankings and three of the six beat Yahoo’s rankings when comparing the top 100 players from each.  In a separate test, two of the six beat Basketball Monster when comparing the top 100 players. Surprisingly, the top performing algorithm was the SHAW-Z ranking, which is most like the oft-criticized method that the other platforms use. The fact that mine came out on top attests to the novel treatment of percentages. The second most successful algorithm was the SHAW-Scarcity-Boosted-mm ranking, a model based on min-max scaling that adds a modest bonus for players that outperform in scarce categories. Although I don’t compare my rankings to Rosenof’s G-Score ranking method, I note that we come to opposite conclusions regarding the value of high performers in scarce categories. 

### Growth/Next Steps
As a research and data science project, I could not be happier. My rankings beat the competition, but I also hope that by publishing the actual ranking methods AND their systematic comparisons that this project inspires further research and discussion. Still, a lot more work can be done here, in terms of metric construction, comparison, and especially on the front-end.

<br>

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

The code above returns a dataframe of 66 columns and 550+ rows, which get added every time a new players sees the court. We're interested in the following columns: 

```python

stats = nba[['PLAYER_NAME', 'GP', 'FGA', 'FGM', 'FTA', 'FTM','FG3A', 'FG3M', 
             'REB', 'AST', 'TOV', 'STL', 'BLK', 'PTS', 'DD2', 
             'FG_PCT', 'FG3_PCT', 'FT_PCT']].copy()

```

A select sample of the data, showing three superstars and one lesser-known player. 

| **Player Name** | **GP** | **FGA** | **FGM** | **REB** | **AST** | **TOV** | **STL** | **BLK** | **PTS** |
|-----------------|--------|---------|---------|---------|---------|---------|---------|---------|---------|
| LeBron James | 62 | 1139 | 583 | 506 | 525 | 238 | 58 | 35 | 1521 |
| Nikola Jokic | 63 | 1226 | 706 | 806 | 647 | 206 | 109 | 43 | 1845 |
| Victor Wembanyama | 46 | 857 | 408 | 506 | 168 | 149 | 52 | 176 | 1116 |
| Jordan Goodwin | 20 | 104 | 50 | 77 | 29 | 16 | 23 | 11 | 130 |



# Transformations

In a typical fantasy basketball league, a team’s weekly score is the sum of players’ cumulative stats in each of the following 9 categories: 
- Field Goal Percentage (FG_PCT = FGM/FGA)
- Free Throw Percentage (FT_PCT = FTM/FTA)
- Points (PTS)
- 3-point Field Goals made (FG3M)
- Rebounds (REB)
- Assists (AST)
- Steals (STL)
- Blocks (BLK)
- Turnovers (TOV) (which count negatively) 


From there, leagues can keep score in one of two ways: a weekly ‘head-to-head’ matchup win (for the team that wins the most categories), or a win for each category (with all nine categories at stake). Leagues can be customized to include more or less categories, but the default is nine, and the main platforms’ rankings are based on these categories. 

The goal is to turn raw NBA statistics from the relevant categories into a single player ranking metric that will help team managers understand a single player’s relative fantasy value. It would be easy if there was one statistical category to keep track of – players would be ranked by the number of points they get, for example. [In fact, *Points* leagues are an alternative scoring format that assigns every statistical category a uniform ‘point’ value, which is too easy to be any fun.] But category leagues require comparisons across 9 different measures. How can you value a player that gets a lot of points and few rebounds vs. one that gets a lot of rebounds and few assists? How do you compare a player that shoots an average free throw percentage on high volume, to a player that shoots above average on low volume?

Before we get into that, it is important to understand the distributions. 


### Distributions

Most NBA stat categories are positively skewed. That is, few players get the most points, rebounds, assists, etc. 

![alt text](/img/posts/PTS_REB_raw.png "Raw Distributions, Points & Rebounds")

At the same time, a large number of players accumulate very little, if any, stats at all. THIS fact is important, although it is hardly considered among the experts. When we start to transform distributions, there is a huge difference in comparing the fantasy-relevant players vs. the entirety of the league. That is, the relative value of a player for fantasy purposes should be considered only in relation to other plausible options, and NOT in relation to the bulk of players that get little or no playing time. 

Before making any transformations, thus, a cut-off should be levied, and I define mine at the 25th percentile of games played. This has the effect of eliminating the ‘garbage time’ players as well as those with season defining injuries (e.g., Embiid in 24-25). Then, I scale by games played. Most fantasy relevant players will miss some time during a season due to minor injuries, so using per-game statistics (which is standard) helps to level the field. 
The result is a more modest 400 + player pool, with the large number of zeros eliminated. 

```python

# Set Games Played threshold for counting stats
minimum_games_threshold_quantile = 0.25
min_games = stats["GP"].quantile(minimum_games_threshold_quantile)

pg_stats = stats[stats["GP"] >= min_games].copy()

raw_categories = ['FGA', 'FGM', 'FTA', 'FTM', 'FG3M', 'PTS',
             'REB', 'AST', 'STL', 'BLK', 'TOV']

for cat in raw_categories:
    pg_stats[cat + "_pg"] = pg_stats[cat] / pg_stats['GP'] 

```

![alt text](/img/posts/PTS_REB_PG.png "Per-Game Points & Rebounds")


## Percentages

Percentage distributions need to be treated differently because they are a function of two distributions: makes and attempts. We don’t simply add percentages like we do the other categories: we divide total team makes by total team attempts to get a final percentage score. A player that shoots an average percentage on high volume of attempts, has a larger impact on a fantasy matchup than an above average shooter that rarely shoots. To evaluate a player’s value in a percentage category, thus, shot volume needs to be considered alongside percent made. I illustrate using Free Throws.

![alt text](/img/posts/FT_PCT_vs_A.png "Free Throw Distributions")

The recieved method for doing this is to measure a player's *impact* by finding the difference between their number of makes and what is expected given their number of attempts and league percentage averages. Standardizing the impact thus gives a comparable value score for that category. For example: 

```python

league_pct = stats["FTM"].sum() / stats["FTA"].sum()
FT_impact = stats["FTM"] - (stats["FTA"] * league_pct)

```

At first glance this would seem fair, a player’s percentage impacts a team’s total to the extent that they take above or below average shot attempts. But because attempts are positively skewed, and percentages are negatively skewed, this method can produce some extreme numbers in both tails. 

![alt text](/img/posts/FT_Impact_Z.png "Free Throw Distributions")

The test case here is Giannis Antetokounmpo. The league average Free Throw percentage (among eligible players) is 78.1% Antetokounmpo shoots a sub-par 60% from, AND he takes the most attempts (he is an elite scorer otherwise, so he gets fouled a lot). The league average Free Throw percentage (among eligible players) is 78.2% Giannis's Free Throw impact Z-Score is -8. 

| **Player Name** | **FT%** | **FTM** | **FTA** | **FT_Impact_Z_Score** |
|-----------------|---------|---------|---------|-----------------------|
| Giannis Antetokounmpo | 60.2 | 369 | 613 | -8.12 | 
| Steph Curry | 92.9 | 252 | 234 | 2.73 | 
| Shae Gilgeous-Alexander | 90.1 | 563 | 625 | 5.50 | 
| James Harden | 87.4 | 515 | 450 | 3.51 | 
| LeBron James | 76.8 | 289 | 222 | -0.23 | 

A critical question for fantasy category valuations is thus, does Giannis hurt you THAT much? Does Shae Gilgeous Alexander *help* you that much? 

For other, counting statistics, I will argue that skewed distributions are meaningful, but - and call it a hunch - although percentage and attempt distributions vary similarly, a percentage statistic is bound between 0 and 1, so positive and negative constributions to it are limited. Realistically, a week-to-week Free-Throw Percentage for an average fantasy team will orbit the league average, plus or minus 0.10% 

## SHAW Percentage Transformation

Rather than standardizing impact, I apply a more modest, sigmoidal weighting formula to attempts, 

that is muted around the mean (of attempts) and only starts to proportionally affect high volume shooters at the positive tail of the attempts distribution. Lambda, which defines the shape of the sigmoid curve, is found dynamically in relation to the skew of the attempts distribution. This ends up being a low number, which 

The result is that Percentage category player valuations 

[IMAGE: Sigmoidal Transformation Formula] 

[IMAGE: Sigmoidal Transformation Graph]

[IMAGE: Graph raw percentages, raw attempts, sigmoid-weighted deficits]. 

[PYTHON CODE BLOCK] 

The resulting distribution is the sigmoid-attempts-weighted deficit, which could be added back to a players raw percentage, or left alone, because the resulting distribution will be the same. 

[IMAGE: TABLE: Giannis, Curry, Shae, LeBron WITH Shaw percentage transformation]

## Standardization

Standardization works because it puts all category distributions in the same theoretical range so that they can be easily compared. A player’s standardized-, or Z-score is their relative difference (above or below the mean) divided by the standard deviation of the distribution.

[mathematical formula] 

In Python: [CODE BLOCK]

And that operation leads to the following: 

[IMAGE:  standardized points/rebounds per game] 
As we can see the spread remains the same AND now the ranges are congruent. From here, we can compare points, rebounds, and SHAW transformed percentages in like terms. AT a Z-score of ~3, 28 Points per game is comparatively equivalent to 11 rebounds per game.
The SHAW-Z ranking is simply the sum the standardized categories, after SHAW-transforming percentages. 
[PYTHON CODE CHUNK]
Min-Max Scaling
Similar to standardization, min-max scaling (transforming the range to 0 – 1) preserves the spread, but it also limits the range, so that outlier values are not so extreme. 
[IMAGE: Math formula]
[IMAGE: min-max points/ rebounds]. 
As we can see, scaling between 0 and 1 produces the same relative distribution, and this time the range IS the same across categories: The highest achiever in each can only see a maximum score of 1. This method seems to limit an outlier players’ advantage in any given category. A maximum of one point for one category seems like a fair system. 
The SHAW-mm ranking is found by summing the normalized (0-1) categories, after SHAW transforming percentages. 
[PYTHON CODE CHUNK]
Scarcity Ranking
Scarcity is the basis of modern economics because scarce resources are worth more than abundant ones. In an NBA game, there are plenty of points scored and many players accumulate points. By contrast, blocked shots or steals might happen a handful of times in a game, and only a few players across the league tend to excel in those categories. 
Although Blocks and Points count the same in terms of categories, having a player that excels in Blocks may be more valuable than a high point getter, because the shot blocker is harder to replace. There are fewer elite shot blockers in the league, and if you don’t have one, you’ll have a hard time competing in that category. 
To test this hypothesis, I developed an index that weighs the relative scarcity of each of the seven cumulative categories (on a scale of 0-1, total scarcity = 1) by subtracting the skew from the inner-quartile range, and normalizing (min-max scaling) the results. Then, for the min-max transformed categories, I multiply the normalized category distributions by its scarcity score. Because both the scarcity index and normalized distributions range between 0-1, the resulting sum of scarcity weighted scores also range between 0 and 1. The result is a modest addition that boosts a player’s min-max score by a maximum of one point. That should be enough to redistribute the players to test whether rewarding scarcity actually improves the rankings. 
IMAGE: SCARCITY MEASURE
[PYTHON CODE CHUNK]
Other Ranking Methods
Other plausible ranking methods use the SHAW transformed percentages but forgo additional transformations. 
Ranked Sum of Category Ranks
The simplest method is to rank players in each category, add up those ranks, and reverse rank that sum. 
[IMAGE: TABLE: 5 Notable players]
This method does not preserve the relative spread, but instead distributes players uniformly in each category, while still accounting for the relative position of each player in each category. 
And since we’re comparing all the players across all categories, this method would seem to be the most efficient. The results are somewhat surprising. Is Christian Braun is above average in most categories, but not elite in any. Is Braun really more valuable than a player that is elite in 4 or five categories, but only average in 5 others?  
Head to Head individual player comparisons
Another approach to ranking involves observing how players match up head-to-head (H2H) against other players. After SHAW-transforming the percentages, players are compared against every other player in every category. This requires building a data frame with a row for each player combination. From there we can count the number of categories that each player wins versus each other, we can assign a matchup winner (for most categories won), and we can count the total categories, and total matchup wins against the field. 
This takes considerable computing time, so it’s not a practical method to build into an ETL pipeline. But I do construct the rankings based on H2H matchup wins and H2H each category wins, and compare these to the other metrics. 
For brevity, I include most of code for this simulation in my github repository, and for consistency, I construct the rankings below, but ONLY after transforming the percentage categories. 
[PYTHON CODE BLOCK]
[IMAGE: TABLE H2H CATEGORIES] 

The six different ranking methods produce a lot of similar rankings, but enough variation to be meaningfully different. 
[TABLE – select players in different rank orders] 








The Competition
In addition to comparing my own ranking metrics, I compare to the rankings of Yahoo, ESPN, and Basketball Monster. 
I mentioned earlier that there is no transparency as to how these platforms compute rankings, and to be fair, Yahoo and ESPN’s fantasy player portals don’t actually include ‘rank’ numbers after the season begins. Nevertheless, they do provide an ostensibly rank-ordered list that can be sorted based on season totals, per-game averages, two week averages, etc. Because the order changes based on the parameter, and because the top players are congruent with other rankings, it is safe to assume there is a ranking algorithm working behind the scenes. 
I copy/pasted the top 200 players for season averages (a.k.a. per-game) statistics on March 26, 2025 from both ESPN and Yahoo. I mention the date because this is a single point-in-time comparison. I refreshed my own rankings on March 26, so that I’m comparing player rankings using the same player statistics.
ESPN provides a separate ‘updated category rankings’ list every week, but these are based on future projections – designed to help fantasy managers make replacement decisions  -and they are different from the lists provided in their fantasy player portal. Still, it does appear that ESPN uses some type of injury forecast, even for their “season averages” list. 
Why Victor Wembanyama was listed by ESPN at the 111th position on their list, however, is beyond me. He should either be a top 10 player by season averages, or he should be completely removed (for forecasting purposes) because he suffered a season-ending blood condition.
Nevertheless, to make the comparisons fair, I removed all the currently injured players in ESPN’s top 130 from the pool of eligible players. That injury list includes: 
[INJURY LIST]
Note that I kept both Nikola Jokic and Anthony Davis, two top players that were scheduled to return from injuries that very day. 
With those indiscrepancies out of the way, there were also a few players in Yahoo’s and ESPN’s average rankings that were filtered out of my own player pool because they did not meet my minimum games threshold. These include Joel Embiid, who has only played X games this season, and [X] a player whose playing time increased only towards the end of the season. Although these players had ranks elsewhere, they were not in the eligible player pool and thus were removed from rank comparisons. 
I compared Basketball Monster rankings separately. Due to a mistake on my part, I did not scrape their rankings until 3/28/2025. Since two more days of NBA games had passed, I decided to compare these separately, and refreshed my own rankings on 3/28/2025 to compare them. Basketball Monster keeps a very clear ranking system that appears to be true to real “season averages”. I did not have to exclude any injured players, but players that did not meet the minimum games threshold were removed by default. 
Top N Players
First, I simulate head-to-head, 9-category matchups using the top n players in each ranking system. I compare the top 20, top 50, top 100, and top 130 (the number of active players in a 10-team league) in separate matchups using real, up-to-date, per-game statistics. 
[PYTHON CODE CHUNK]
There is of course overlap between the metrics, but there is enough difference to put the algorithms to the test. For example, a punishing -6 (standard attempt weighted) Z-score in the free throw percentage category would put Giannis out of the top 20, while a modest -3 (SHAW weighted) Z-score might ensure that he remains. If Giannis were excluded from the upper lists, he would likely be reincluded by the time we reach top 100 and top 130 players. As n gets larger, however, we should expect there to be more variation between ranking systems.
IMAGE Results table. 
The results of this experiment seem clear, if not conclusive. The top performing ranking system is…. X, 
Yahoo, and Basketball Monster also offer competitive rankings. ESPN’s rankings don’t match up well, and this is likely due to the confusion (and lack of transparency) about what ESPN’s rankings actually mean. They appear to give extraordinary weight to a player’s short-term statistics, which is GREAT if you are looking for a replacement player at any point in time. Still, when toggling ‘all players’ and ‘season averages’ this is the ordered list that they provide. 

Snake-Draft Tournament
A second method of comparing rankings is to simulate 10 teams from each, which should provide an approximately average team from that ranking metric. But the results of head-to-head matchups among the scores of teams using this method offer a drastically different picture. 
[IMAGE: Top and bottom 10 teams]. 
Given the results of the top-n teams, we might expect that the SHAW-Z and SHAW-Scarcity-Boosted-mm ranked teams would rise to the top. Indeed these metrics take the top to matchup-wins spots, but the Basketball Monster rankings get the most total category wins. Perhaps more interestingly, the differences among the top performers appear insignificant. Shaw-Z wins only 1% more than the next. 
[IMAGE: Cumlative ranking results]
Unpacking the team team-level standings further, we see that same ranking systems perform across the range of wins and losses. In fact, the H2H rankings are among the top winners (and losers) when they did not perform well at all in top-n teams matchups. This experiment leads me to think that – all other things being equal – the luck of the draw – in this case the draft order given slightly different combinations of players – matters the more than rankings themselves. 
Ranking Matchup Summary:
I am proud to say that many of my rankings outperformed the competition in head-to-head matchups. Specifically, if my Z-ranking beat other rankings that are based on standardization (specifically Yahoo and BBM), the difference can be found in the novel way that I treat percentage transformations. 
Additionally, the relative success of my ‘scarcity boosted’, min-max ranking should demonstrate that scarcity does matter, and this finding should inform future improvements. 
It may also be worth noting the limitations of my experiments. The simulated tournament revealed the surprising finding that, when dividing any rank-based player pool into n teams, the variances that made the whole cohere become unstable, and draft order and team build might weigh more heavily. In fact, my snake-draft simulation did not include player position rules, which are an integral part of fantasy team building. It is likely that, for each ranking metric, the snake draft produced a set of teams where certain positions (and thus certain statistical contributions) were over or underrepresented. 
Value over Replacement
I construct one final metric for my player-ranker dashboard, a VORP score – which I construct in two forms – and which is simply a player’s SHAW-Z score or SHAW-mm score in relation to the average of the 100-130 ranked players. 
Because Z-scores rankings reward high achievers in specific categories, and because Min-Max scores reflect balance across categories (i.e., no one category over shadows another), I rebrand these “Impact Value” and “Balance Value”, respectively, and give the user a change to evaluate players in those terms for their unique team needs. 
[PYTHON CODE CHUNK]

Load / Endpoint / Streamlit App
Finally, as a proof of concept, I build a Streamlit App as a user endpoint. The app allows the user to select among my top ranking metrics (except the H2H ones that take computing time). 
It is not yet deployed outside of my local drive, but my near-future goal is to make this publically accessible, executing the whole ETL pipeline described above. 


I.	Conclusion/Growth

think it is safe to conclude that rankings only make up a small part fantasy success. Even when using the most competitive method (z scores), there is still intransitivity among players, and then the draft order generates noise that rankings may not be able to overcome. 
Even more confounding would be the actual position requirements in fantasy leagues, which my simulated matchups do not account for. Savvy fantasy managers know as well that building a team is equally art and science. Knowing that you only need 5 of 9 categories to win, for example, you can elect to ‘punt’ or exclude one or more categories from your ideal team build. Knowing which categories to punt depends on what players are available at what order during draft day. A ranking system that accounted for every punt scenario in real time during a player draft would be overwhelming to say the least. 
And finally, I have not addressed factor that all fantasy managers know to be the one the real difference maker basketball – injuries. This year, perennial MVP favorite Nikola Jokic missed 5 straight games during fantasy playoff weeks. If you had him on your team, you were likely in the playoffs, and then you likely lost because he was injured. 

COMMENTS ON STANDARDIZATION and its CRITICS. 
The criticism that Z-scores over-value outliers in skewed distributions turns out to be wrong. My standardized rankings beat all the other rankings, but WHY? It turns out that standardization, like my scarcity boost formula, rewards scarcity. When a distribution is highly skewed, that may be because the event is scarce, or because elite producers in that category are scarce. In fact, eliteness produces skew, and thus scarcity. Victor Wembanyama pushes out the tail of the distribution because he is a unique talent; the skew results from the fact that there is no one else out there with him. Wembanyma’s Z-score of 8 in Blocks, or Dyson Daniel’s Z-score of 6 in Steals not only reflect those player’s ability in relation to those categories, but they reflect their scarcity in relation to other players as well. 
To say that standardization assumes a normal distribution, is like saying wearing running shoes assumes that I am running. The misconception comes from the statistical concept of the central limit theorem, which allows us to assign probabilities to independent events occurring because of the fact that sample distributions ARE normally distributed when numbers are large. Then, standard deviations have real consequences. But we know that NBA stats aren’t normal, and we are not trying to test whether or not Wemby’s 3.5 blocks per game comes from the same distribution as LeBron Jame’s [X] Blocks. We already know they do. I am however, interested in knowing how Wembanyama’s 3.5 blocks per game compares to Sabonis’ 14 rebounds, or Trae Young’s 10+ assists. Standardization simply gives us a like metric to compare categories that otherwise have radically different ranges. The fact that it produces some astounding numbers at the tails is actually a benefit for comparing cumulative fantasy categories when the few players that can achieve those results can only be picked once. 

 
The criticism of this approach is that standardization is best applied when distributions are normal because standard deviation units represent real theoretical probabilities. A standard deviation of 3 in a normal distribution theoretically represents the top 99.7 percentile. In a league of 500+ players, we should expect to see z-scores of 3 or higher about two times in each category. A z-score of 5 or more would be theoretically implausible. 
But NBA stats are NOT normal distributions, so standardizing seems to produce some absurd numbers. Can we really compare Victor Wembanyama’s z-score of 7 in the Blocks category to the highest point getter (4?) The categories count the same, but Wemby seems to get two categories worth of points. 
One of the reasons that standardization seems to work even though it produces some obscene numbers is because z-scores reward scarcity. That is, they value extraordinary players in proportion to the extent that they dominate. 

Table Ranking Tournament Matchup and Category Wins vs. ESPN & YAHOO 3/26
print(aggregated_wins)
                     Matchup_Wins  Category_Wins
Metric                                          
9_cat_z_rank                  423           3636
9_cat_scarcity_rank           414           3641
ESPN Rank                     407           3549
9_cat_mm_rank                 405           3574
Yahoo Rank                    401           3605
9_cat_rank_sum_rank           395           3589
H2H_9_most_rank               374           3444
H2H_9_each_rank               341           3402

Table Ranking Tournament Matchup and Category Wins vs. BBM 3/28

print(aggregated_wins)
                     Matchup_Wins  Category_Wins
Metric                                          
9_cat_scarcity_rank           368           3184
bbm_ranks                     364           3196
9_cat_z_rank                  361           3175
H2H_9_each_rank               359           3130
9_cat_mm_rank                 353           3139
9_cat_rank_sum_rank           317           2988
H2H_9_most_rank               293           2923




