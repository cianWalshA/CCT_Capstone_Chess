# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 20:40:31 2023

@author: cianw
"""
import sys
import io
import ast
import os
import time
import csv
import chess
import chess.pgn
import stockfish
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from datetime import datetime 


stockfish_Path = Path(r"C:\Users\cianw\Chess Engines\Latest\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")
lc0_Path = Path(r"C:\Users\cianw\Chess Engines\Latest\lc0-v0.30.0-windows-gpu-nvidia-cuda\lc0.exe")
 
pgnFolder = r"E:\ChessData"
csvFolder = r"E:\ChessData"
csvAllRatingFolder = r"E:\ChessData\newOutputs"

pgnName = "lichess_db_standard_rated_2023-06_2000_1m_row_analysed"
pgnIn_EnglineAnalysis = Path(rf"{csvFolder}\{pgnName}.tsv")

pgnNameAllRating = r"lichess_db_standard_rated_2023-06__allRatings6GB"
pgnIn_AllRatings = Path(rf"{csvAllRatingFolder}\{pgnNameAllRating}.csv")

"""
lichessData = pd.read_csv(pgnIn_EnglineAnalysis, sep='\t', nrows=100000)
lichessData['UTC_dateTime'] = pd.to_datetime(lichessData['UTCDate'] + ' ' + lichessData['UTCTime'])
lichessData.describe()
"""

allRatings = pd.read_csv(pgnIn_AllRatings)
allRatings['UTC_dateTime'] = pd.to_datetime(allRatings['UTCDate'] + ' ' + allRatings['UTCTime'])
allRatings.describe()

openings_a_Path = Path(rf"{csvFolder}\a.tsv")
openings_b_Path = Path(rf"{csvFolder}\b.tsv")
openings_c_Path = Path(rf"{csvFolder}\c.tsv")
openings_d_Path = Path(rf"{csvFolder}\d.tsv")
openings_e_Path = Path(rf"{csvFolder}\e.tsv")
openings_a = pd.read_csv(openings_a_Path, sep='\t')
openings_b = pd.read_csv(openings_b_Path, sep='\t')
openings_c = pd.read_csv(openings_c_Path, sep='\t')
openings_d = pd.read_csv(openings_d_Path, sep='\t')
openings_e = pd.read_csv(openings_e_Path, sep='\t')
openings = pd.concat([openings_a, openings_b, openings_c, openings_d, openings_e])
del openings_a, openings_b, openings_c, openings_d, openings_e


"""
SECTION 0 - 
Functions
"""
#Function to Extract Every Nth Word of a string delimted by spaces starting at position M, up to N*O words.
def extract_nth_words(text, M, N, O=None):
    words = text.split()
    if O is None:
        endIndex = len(words)
    else: 
        endIndex = min(M-1+N*O, len(words))
    result = [words[i] for i in range(M - 1, endIndex, N)]
    return ' '.join(result)

#Function to return every ith element of a list that was saved as a string
def get_ith_element(lst, i):
    res = lst.strip('][').split(', ')
    if len(res) >= i+1:
        return res[i]
    else:
        return None

#Incorporate get_ith_element into this function
#Return X moves with X details, save double defining the ouptuts

def move_analysis(text, M, N, O=None):
    
    #Call word extractor based on list
    moves = extract_nth_words(text, M, N, O)
    
    #Process Moves into Features
    checks = moves.str.count('+')
    takes = moves.str.count('x')
    bishopMoves = moves.str.count('B')
    knightMoves = moves.str.count('N')
    rookMoves = moves.str.count('R')
    
    #Pawn Moves - Need to clarify - pseudo code below
    #Words in string that start with a small latter - regex -> a-z
    
    #Repeated Moves - Where a word in string - check for repetition of it in X moves
    
    minorMoves = bishopMoves + knightMoves + rookMoves
    return

def summarize_columns(df, groupCols, prefixes, summaryStats):
    grouped = df.groupby(groupCols)
    summary_df = pd.DataFrame()

    for prefix in prefixes:
        for stat in summaryStats:
            selectedCols = [col for col in df.columns if col.startswith(prefix)]
            print(selectedCols)
            print(stat)
            print(summaryStats)
            result = grouped[selectedCols].agg({stat: rf'{stat}'})
            result.columns = [f'{prefix}_{col}_{stat}' for col in selectedCols]
            summary_df = pd.concat([summary_df, result], axis=1)
    summary_df = summary_df.reset_index()
    
    return summary_df

def normal_test_subgroups(df, dataCol, subgroup , alpha):
    results=[]
    uniqueVals = pd.Series.unique(df[subgroup])
    for value in uniqueVals:
        subgroup_df = df[df[subgroup]==int(value)]
        
        #Apply Shapiro-Wilk Normality Test
        tStat, pValue = stats.shapiro(subgroup_df[dataCol])
        # Determine if the subgroup follows a normal distribution
        normal = pValue > alpha
        
        results.append((value, tStat, pValue, normal))
        print(rf"Variable: {dataCol} - Subgroup: {value} - test: {tStat} - p-Value: {pValue} - Normal: {normal}")
    return pd.DataFrame(results, columns = [subgroup, 'testStatistic', 'pValue', 'isNormal'])



"""
lichessData = lichessData.join(pd.DataFrame(lichessData['SF_eval'].apply(ast.literal_eval).values.tolist()).add_prefix('eval')).drop(columns={'SF_eval'})
lichessData = lichessData.join(pd.DataFrame(lichessData['SF_seldepth'].apply(ast.literal_eval).values.tolist()).add_prefix('seldepth')).drop(columns={'SF_seldepth'})

lichessData = lichessData.dropna(subset='eval4')

selectedCols = [col for col in lichessData.columns if col.startswith('eval')]
lichessSummary = lichessData.groupby('Opening_x')[selectedCols].describe()
lichessSummary.columns = [' '.join(col).strip() for col in lichessSummary.columns.values]

"""



"""
Section XYZ - Feature Extraction from Complete Set

lichessData['whiteMoves_5'] = lichessData['Moves'].apply(lambda x: extract_nth_words(x, 2, 3, 5))
lichessData['blackMoves_5'] = lichessData['Moves'].apply(lambda x: extract_nth_words(x, 3, 3, 5))
lichessData['whiteMoves_10'] = lichessData['Moves'].apply(lambda x: extract_nth_words(x, 2, 3, 10))
lichessData['blackMoves_10'] = lichessData['Moves'].apply(lambda x: extract_nth_words(x, 3, 3, 10))
lichessData['whiteMoves_20'] = lichessData['Moves'].apply(lambda x: extract_nth_words(x, 2, 3, 20))
lichessData['blackMoves_20'] = lichessData['Moves'].apply(lambda x: extract_nth_words(x, 3, 3, 20))
lichessData['whiteMoves_30'] = lichessData['Moves'].apply(lambda x: extract_nth_words(x, 2, 3, 30))
lichessData['blackMoves_30'] = lichessData['Moves'].apply(lambda x: extract_nth_words(x, 3, 3, 30))

lichessData['whiteChecks'] = lichessData['whiteMoves'].str.count('+')
lichessData['blackChecks'] = lichessData['whiteMoves'].str.count('+')
lichessData['whiteTakes'] = lichessData['whiteMoves'].str.count('x')
lichessData['blackTakes'] = lichessData['whiteMoves'].str.count('x')
"""

#Count and plot how many games based on move length

#Count and plot how many games based on each opening



#Where move count >= 5, >=10 return 10th and 20th values of rating respectively?

"""
SECTION - ABC
Exploration of rating distribution
"""
#Get key information from openings dataframe
openings['moveNumbers'] = openings['pgn'].apply(lambda x: extract_nth_words(x, 1, 3))
openings['whiteMoves'] = openings['pgn'].apply(lambda x: extract_nth_words(x, 2, 3))
openings['blackMoves'] = openings['pgn'].apply(lambda x: extract_nth_words(x, 3, 3))
openings['moveCount'] = openings['moveNumbers'].str.split().str.len()
openings['halfMoveCount'] = openings['whiteMoves'].str.split().str.len() + openings['blackMoves'].str.split().str.len()  
openings['white_black'] = openings['halfMoveCount'].apply(lambda x: 'black' if x % 2 == 0 else 'white') 
# Sort the Openings by their name and moves involved, only keep the minimal amount of moves to avoid complexity.
openings = openings.sort_values(by=['name', 'halfMoveCount'])
openings = openings.drop_duplicates(subset='name', keep='first')

#Only Working with white openings - 
openings = openings[openings['white_black']=='white']

#Extraction of opening structure
openings[['Basis', 'lineVariation']] = openings['name'].str.split(': ', 1, expand=True)
openings['Line'] = openings['lineVariation'].str.split(',').str[0]
openings['Variation'] = openings['lineVariation'].str.split(',',1).str[-1]
openings['Variation'] = np.where(openings['Variation']==openings['Line'], None, (openings['Variation'].str.strip()))
openings = openings.drop(columns=['moveNumbers', 'whiteMoves' , 'blackMoves', 'lineVariation'])

#Drop Variation games, too much granularity
openings = openings[pd.isna(openings['Line'])]

#Adding opening information to chess games dataset
allRatings = allRatings.merge(openings, left_on='Opening', right_on='name', how='inner')
allRatings['openingPlayer'] = np.where(allRatings['white_black'] == 'black', allRatings['Black'], allRatings['White'])
allRatings['openingPlayerRating'] = np.where(allRatings['white_black'] == 'black', allRatings['BlackElo'], allRatings['WhiteElo'])

allPlayers= pd.DataFrame(allRatings['openingPlayer'].unique()).reset_index()

openingsPlayed = allRatings.groupby('Opening').size().reset_index(name='timesPlayed')
uniquePlayers = allRatings.groupby('Opening')['openingPlayer'].nunique().reset_index().rename(columns={'openingPlayer': 'uniquePlayers'})
openingAnalysis = openingsPlayed.merge(uniquePlayers, on='Opening')
openingAnalysis['openingPlayerDiversity'] =1-(openingAnalysis['timesPlayed']-openingAnalysis['uniquePlayers'])/openingAnalysis['timesPlayed']


uniquePlayerOpenings = allRatings.groupby('openingPlayer')['Opening'].nunique().reset_index().rename(columns={'Opening': 'countPlayerOpening'})
uniquePlayerGames = allRatings.groupby('openingPlayer').size().reset_index(name='countGames')
playerAnalysis = uniquePlayerOpenings.merge(uniquePlayerGames, on='openingPlayer')
playerAnalysis['openingsUsedDivesity'] =1-(playerAnalysis['countGames']-playerAnalysis['countPlayerOpening'])/playerAnalysis['countGames']

playerOpenings =  allRatings.groupby(['openingPlayer', 'Opening']).size().reset_index(name='openingUseCount')
playerOpenings['openingProbability'] = playerOpenings.groupby('openingPlayer')['openingUseCount'].apply(lambda x: x / float(x.sum()))# Calculate entropy for each player

entropy = playerOpenings.groupby('openingPlayer').apply(lambda x: -np.sum(x['openingProbability'] * np.log2(x['openingProbability'])))

# Calculate Gini coefficient for each player
gini = playerOpenings.groupby('openingPlayer').apply(lambda x: 1 - np.sum(x['openingProbability']**2))

# Calculate Concentration Index for each player
concentrationIndex = playerOpenings.groupby('openingPlayer').apply(lambda x: 1 - 2 * np.sum((x['openingProbability'].cumsum() - x['openingProbability'] / 2)))

# Combine results into a DataFrame
diversity = pd.DataFrame({
    'entropy': entropy,
    'gini': gini,
    'ConcentrationIndex': concentrationIndex
}).reset_index()

playerAnalysis = playerAnalysis.merge(diversity, on='openingPlayer', how='left')


allRatingsExp = allRatings.merge(openingAnalysis, on='Opening', how='left')
allRatingsExp = allRatingsExp.merge(playerAnalysis, on='openingPlayer', how='left')

allRatingsRatioFilter = allRatingsExp[(allRatingsExp['uniquePlayers']>100) &
                                      (allRatingsExp['openingsUsedDivesity']>=0.2) &
                                      (allRatingsExp['openingPlayerDiversity']!=1) &
                                      (allRatingsExp['countGames']>=10)
                                      ]
"""

allRatingsRatioFilter = allRatingsExp[(allRatingsExp['countPlayerOpening']!=1)&
                                      allRatingsExp['countGames'] != allRatingsExp['countPlayerOpening'] &
                                      (allRatingsExp['countGames']>=10)]
"""

playersForAnalysis = allRatingsRatioFilter['openingPlayer'].unique()


whiteElo = allRatingsRatioFilter[['White','WhiteElo']].rename(columns={'White':'openingPlayer', 'WhiteElo':'ELO'})
blackElo = allRatingsRatioFilter[['Black','BlackElo']].rename(columns={'Black':'openingPlayer', 'BlackElo':'ELO'})

playerRatingsBoth = pd.concat([whiteElo,blackElo])
playerRatings = playerRatingsBoth.groupby('openingPlayer')['ELO'].mean().reset_index(name='ELO')
playerRatings = playerRatings.merge(playerAnalysis, on='openingPlayer', how='left')
playerRatings = playerRatings.dropna(subset=['openingPlayer'])
playerRatings = playerRatings[playerRatings['openingPlayer'].isin(playersForAnalysis)]

"""
Calculate Shannon Diversity Index and use for tests
"""
"""
player_groups = allRatingsRatioFilter.groupby('openingPlayer')['Opening'].value_counts().unstack(fill_value=0)
total_games_played = player_groups.sum(axis=1)
# Calculate the probability distribution of openings for each player
probability_distribution = player_groups.div(total_games_played, axis=0)
# Calculate Shannon's entropy for each player
shannon_entropy = -np.sum(probability_distribution * np.log2(probability_distribution), axis=1)
# Create a DataFrame with player diversity information
player_diversity = pd.DataFrame({'openingPlayer': shannon_entropy.index, 'Diversity': shannon_entropy}).reset_index(drop=True)

playerRatings = playerRatings.merge(player_diversity, on='openingPlayer', how='left')
"""
playerRatings = playerRatings.sort_values(by=['ELO']).reset_index(drop=True)
splits=3
playerRatings['bins'] = pd.cut(x=playerRatings['ELO'], bins=splits, labels=np.arange(1,splits+1,1))

alpha = 0.05
tStat, pValue = stats.shapiro(playerRatings['openingsUsedDivesity'])
normal = pValue > alpha
print(rf"Test Statistic: {tStat} - P-Value: {pValue} - Normal: {normal}")

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Select relevant columns
selected_columns = ['ELO', 'openingsUsedDivesity']

# Feature engineering (if needed)
# Normalize or standardize the features
scaler = StandardScaler()
playerRatingsT = pd.DataFrame()
playerRatingsT[selected_columns] = scaler.fit_transform(playerRatings[selected_columns])

# Combine features into a single feature matrix
X = playerRatingsT[selected_columns]

# Determine the optimal number of clusters (e.g., using the Elbow Method)
wcss = []
for num_clusters in range(1, 11):
    kmeans = KMeans(n_clusters=num_clusters, random_state=1)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method curve to choose the number of clusters
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Choose the optimal number of clusters based on the plot (e.g., from the "elbow" point)

# Perform K-means clustering with the chosen number of clusters
optimal_num_clusters = 3  # Adjust this based on your analysis
kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=0)
playerRatingsT['Cluster'] = kmeans.fit_predict(X)

cluster_means = playerRatingsT.groupby('Cluster')[selected_columns].mean().reset_index()
cluster_means_sorted = cluster_means.sort_values(by=selected_columns[0], ascending=True)  # Replace 'Rating' with the desired feature
cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(cluster_means_sorted['Cluster'])}
playerRatings['Cluster'] = playerRatingsT['Cluster'].map(cluster_mapping)

# Analyze the clusters and divisions in ratings and diversity in openings
cluster_centers = kmeans.cluster_centers_
print("Cluster Centers:")
print(cluster_centers)


sns.scatterplot(data =playerRatings, x=selected_columns[0], y= selected_columns[1], hue = 'Cluster', alpha=0.25 , palette='colorblind')
plt.show()
sns.histplot(data=playerRatings, 
             x=selected_columns[1], 
             hue='bins', 
             bins=50, 
             kde=True, 
             stat="density", 
             palette='colorblind',
             common_norm=False,
             fill=True,
             alpha=0.25)
plt.show()





from scipy.stats import chi2_contingency
# Create a contingency table of openings vs. players with counts
contingency_table = pd.crosstab(allRatingsRatioFilter['Player'], allRatingsRatioFilter['Opening'])

# Perform the Chi-Square Test for Independence
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Check the p-value to determine if the association is significant
if p < 0.05:
    print("There is a significant association between player decisions and openings.")
else:
    print("There is no significant association between player decisions and openings.")

# Display the expected frequencies
expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
print("\nExpected Frequencies:")
print(expected_df)


"""
SECTION - ABC
Exploration of Engine Analysed Sub-Sample
"""



"""
SECTION - DEF
Chess Puzzles Testing for Engines
"""







