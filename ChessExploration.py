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

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

"""
SECTION 0
Paths and Imports
"""
stockfish_Path = Path(r"C:\Users\cianw\Chess Engines\Latest\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")
lc0_Path = Path(r"C:\Users\cianw\Chess Engines\Latest\lc0-v0.30.0-windows-gpu-nvidia-cuda\lc0.exe")
 
pgnFolder = r"E:\ChessData"
csvFolder = r"E:\ChessData\explorationOutputs"
csvAllRatingFolder = r"E:\ChessData\explorationOutputs"
outputFolder = r"C:\Users\cianw\Documents\dataAnalytics\projectFinal\figureOutputs"

pgnName = "lichess_db_standard_rated_2023-06_2000_1m_row_analysed"
pgnIn_EnglineAnalysis = Path(rf"{csvFolder}\{pgnName}.tsv")

pgnNameAllRating = r"lichess_db_standard_rated_2023-06__10MinGames_15Jun2023_limit_10GB"
pgnIn_AllRatings = Path(rf"{csvAllRatingFolder}\{pgnNameAllRating}.csv")

"""
lichessData = pd.read_csv(pgnIn_EnglineAnalysis, sep='\t', nrows=100000)
lichessData['UTC_dateTime'] = pd.to_datetime(lichessData['UTCDate'] + ' ' + lichessData['UTCTime'])
lichessData.describe()
"""

allRatings = pd.read_csv(pgnIn_AllRatings, nrows=10000000)
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
openings = openings.rename(columns={'eco':'ECO', 'name':'Opening'})
del openings_a, openings_b, openings_c, openings_d, openings_e
del openings_a_Path, openings_b_Path, openings_c_Path, openings_d_Path, openings_e_Path


"""
SECTION 0.1 
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


print(1)
#Where move count >= 5, >=10 return 10th and 20th values of rating respectively?

"""
###########################################################################################################################
Section 2
Opening ECO Filtering
###########################################################################################################################
"""


openingVariable = 'Opening'

#Get key information from openings dataframe
openings['moveNumbers'] = openings['pgn'].apply(lambda x: extract_nth_words(x, 1, 3))
openings['whiteMoves'] = openings['pgn'].apply(lambda x: extract_nth_words(x, 2, 3))
openings['blackMoves'] = openings['pgn'].apply(lambda x: extract_nth_words(x, 3, 3))
openings['moveCount'] = openings['moveNumbers'].str.split().str.len()
openings['halfMoveCount'] = openings['whiteMoves'].str.split().str.len() + openings['blackMoves'].str.split().str.len()  
openings['white_black'] = openings['halfMoveCount'].apply(lambda x: 'black' if x % 2 == 0 else 'white') 
#Sort the Openings by their name and moves involved, only keep the minimal amount of moves to avoid complexity of mixed openings
openings = openings.sort_values(by=[openingVariable, 'halfMoveCount'])
openings = openings.drop_duplicates(subset='Opening', keep='first')

#Only Working with white openings
openings = openings[openings['white_black']=='white']

#Extraction of opening structure
openings[['Basis', 'lineVariation']] = openings['Opening'].str.split(': ', 1, expand=True)
openings['Line'] = openings['lineVariation'].str.split(',').str[0]
openings['Variation'] = openings['lineVariation'].str.split(',',1).str[-1]
openings['Variation'] = np.where(openings['Variation']==openings['Line'], None, (openings['Variation'].str.strip()))
openings = openings.drop(columns=['moveNumbers', 'whiteMoves' , 'blackMoves', 'lineVariation'])

#Drop Variation games, too much granularity
openings = openings[pd.isna(openings['Variation'])]

"""
###########################################################################################################################
Section 3
Aggregate Analysis of Openings and Players that use the openings
###########################################################################################################################
"""
openingList = openings['Opening'].unique()

#Adding opening information to chess games dataset
allRatings = allRatings[allRatings['Opening'].isin(openingList)] # Added to help reduce memory stress of join
allRatings = allRatings.merge(openings, on='Opening', how='inner')
allRatings['openingPlayer'] = np.where(allRatings['white_black'] == 'black', allRatings['Black'], allRatings['White'])
allRatings['openingPlayerRating'] = np.where(allRatings['white_black'] == 'black', allRatings['BlackElo'], allRatings['WhiteElo'])
allRatings['whiteWin'] = np.where(allRatings['Result'].str.split('-').str[0] == '1', 1, 0)

#Analysis of Openings used in Lichess database
openingsPlayed = allRatings.groupby(openingVariable).size().reset_index(name='timesPlayed')
uniquePlayers = allRatings.groupby(openingVariable)['openingPlayer'].nunique().reset_index().rename(columns={'openingPlayer': 'uniquePlayers'})
openingAnalysis = openingsPlayed.merge(uniquePlayers, on=openingVariable)
openingAnalysis['useRatio'] = openingAnalysis['timesPlayed']/openingAnalysis['timesPlayed'].sum()
openingAnalysis['openingPlayerDiversity'] =1-(openingAnalysis['timesPlayed']-openingAnalysis['uniquePlayers'])/openingAnalysis['timesPlayed']
print(openingAnalysis.describe())

#Opening Count Number Filter
openingFilter = openingAnalysis[(openingAnalysis['timesPlayed']>=10) & (openingAnalysis['uniquePlayers']>100)].Opening.unique()
allRatings = allRatings[allRatings['Opening'].isin(openingFilter)]


#Analysis of Players used in Lichess database
uniquePlayerOpenings = allRatings.groupby('openingPlayer')[openingVariable].nunique().reset_index().rename(columns={openingVariable: 'countPlayerOpening'})
uniquePlayerGames = allRatings.groupby('openingPlayer').size().reset_index(name='countGames')
playerAnalysis = uniquePlayerOpenings.merge(uniquePlayerGames, on='openingPlayer')
playerAnalysis['openingsAsRatio'] =(playerAnalysis['countPlayerOpening'])/playerAnalysis['countGames']
playerAnalysis['openingsUsedDivesity'] =1-(playerAnalysis['countGames']-playerAnalysis['countPlayerOpening'])/playerAnalysis['countGames']
print(playerAnalysis.describe())

#Games Player Filter
playerFilter = playerAnalysis[(playerAnalysis['countPlayerOpening']>1) &
                             (playerAnalysis['openingsAsRatio']!=1) &
                             (playerAnalysis['countGames']>=10)].openingPlayer.unique()

allRatings = allRatings[allRatings['openingPlayer'].isin(playerFilter)]
"""
###########################################################################################################################
Section 3
#Creation of Openings Used Diversity variables to add to playerAnalysis
###########################################################################################################################
"""


playerOpenings =  allRatings.groupby(['openingPlayer', openingVariable]).size().reset_index(name='openingUseCount')
playerOpenings['whiteWin'] = allRatings.groupby(['openingPlayer', openingVariable])['whiteWin'].sum().reset_index(drop=True)
playerOpenings['openingProbability'] = playerOpenings.groupby('openingPlayer')['openingUseCount'].apply(lambda x: x / float(x.sum()))
playerOpenings['openingWinProbability'] = playerOpenings.groupby('openingPlayer')['whiteWin'].apply(lambda x: x / float(x.sum()))
#playerOpenings['openingWinProbability'] = playerOpenings['whiteWin']/playerOpenings['openingUseCount']

# Calculate Gini coefficient for each player
playGini = playerOpenings.groupby('openingPlayer').apply(lambda x: 1 - np.sum(x['openingProbability']**2))
# Calculate Entropy Index for each player
playEntropy = playerOpenings.groupby('openingPlayer').apply(lambda x: stats.entropy(x['openingProbability']))

# Calculate Gini coefficient for each player
winGini = playerOpenings.groupby('openingPlayer').apply(lambda x: 1 - np.sum(x['openingWinProbability']**2))
# Calculate Entropy Index for each player
winEntropy = playerOpenings.groupby('openingPlayer').apply(lambda x: stats.entropy(x['openingWinProbability']))


# Combine results into a DataFrame
diversity = pd.DataFrame({
    #'entropy': entropy,
    'playGini': playGini,
    #'concentrationIndex': concentrationIndex,
    'playEntropy': playEntropy,
    'winGini': winGini,
    #'concentrationIndex': concentrationIndex,
    'winEntropy': winEntropy
}).reset_index()

playerAnalysis = playerAnalysis.merge(diversity, on='openingPlayer', how='left')


"""
###########################################################################################################################
Section 4
Merging of Games, Player Analysis, Opening Analysis and Diversity Data
Filtering Games by openings played and how diverse they played
###########################################################################################################################
"""

allRatingsExp = allRatings.merge(openingAnalysis, on= openingVariable, how='left')
allRatingsExp = allRatingsExp.merge(playerAnalysis, on='openingPlayer', how='left')



whiteElo = allRatingsExp[['White','WhiteElo']].rename(columns={'White':'openingPlayer', 'WhiteElo':'ELO'})
blackElo = allRatingsExp[['Black','BlackElo']].rename(columns={'Black':'openingPlayer', 'BlackElo':'ELO'})

playerRatingsBoth = pd.concat([whiteElo,blackElo])
playerRatings = playerRatingsBoth.groupby('openingPlayer')['ELO'].mean().reset_index(name='ELO')
playerRatings = playerRatings.merge(playerAnalysis, on='openingPlayer', how='left')
playerRatings = playerRatings.dropna(subset=['openingPlayer'])
playerRatings = playerRatings[playerRatings['openingPlayer'].isin(playerFilter)]

playerRatings.to_csv(rf"{csvFolder}\playerRatings_diversity.tsv", sep='\t')
allRatings.to_csv(rf"{csvFolder}\lichess_games_filtered.tsv", sep='\t')


allRatings = allRatings.merge(playerRatings[['openingPlayer', 'ELO']], on='openingPlayer', how='left')

"""
###########################################################################################################################
Section 5
Clustering Tests - Diversity in player opening Selection
###########################################################################################################################
"""

selected_columns = ['ELO', 'playGini', 'playEntropy']


# Normalize or standardize the features
scaler = MinMaxScaler()
playerRatingsT = pd.DataFrame()
playerRatingsT[selected_columns] = scaler.fit_transform(playerRatings[selected_columns])


# Combine features into a single feature matrix
X = playerRatingsT[selected_columns]

# Determine the optimal number of clusters (e.g., using the Elbow Method)
cluster_range =range(1, 11)
silhouette_scores1 = []
silhouette_clusters1 = []

sse_scores1 = []
sse_clusters1 = []
for num_clusters in cluster_range:
    print(f"Start Cluster: {num_clusters}")
    kmeans = KMeans(n_clusters=num_clusters, random_state=123)
    cluster_labels = kmeans.fit_predict(X)
    
    # Calculate the Sum of Squared Errors (SSE) for the elbow method
    sse = kmeans.inertia_
    sse_scores1.append(sse)
    sse_clusters1.append(num_clusters)
    
    if num_clusters>=2:
        # Calculate the silhouette score
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores1.append(silhouette_avg)
        silhouette_clusters1.append(num_clusters)

# Plot WCSS (Elbow Method) on the left Y-axis
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=600)
ax1.plot(sse_clusters1, sse_scores1, marker='o', label='WCSS')
ax1.set_xlabel('Number of Clusters')
ax1.set_ylabel('WCSS')
#ax1.set_title('Elbow Method and Silhouette Score for Optimal Number of Clusters')

# Create a twin Y-axis on the right for Silhouette Score
ax2 = ax1.twinx()
ax2.plot(silhouette_clusters1, silhouette_scores1, marker='o', label='Silhouette Score', color='orange')
ax2.set_ylabel('Silhouette Score')

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

# Show the plot
plt.grid(True)
plt.show()




# Perform K-means clustering with the chosen number of clusters
optimal_num_clusters = 3 # Adjust this based on your analysis
kmeans = KMeans(n_clusters=optimal_num_clusters)
playerRatingsT['Cluster'] = kmeans.fit_predict(X)

cluster_means = playerRatingsT.groupby('Cluster')[selected_columns].mean().reset_index()
cluster_means_sorted = cluster_means.sort_values(by=selected_columns[0], ascending=True)  # Replace 'Rating' with the desired feature
cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(cluster_means_sorted['Cluster'])}
playerRatings['Cluster'] = playerRatingsT['Cluster'].map(cluster_mapping)

playerRatings.to_csv(rf"{csvFolder}\playerRatings_Cluster.tsv", sep='\t')
# Analyze the clusters and divisions in ratings and diversity in openings
cluster_centers = kmeans.cluster_centers_
print("Cluster Centers:")
print(cluster_centers)


sns.scatterplot(data =playerRatings, x=selected_columns[0], y= selected_columns[1], hue = 'Cluster', alpha=0.25 , palette='colorblind')
plt.show()

fig, axes = plt.subplots(1,1, figsize=(10, 6), dpi=600)
fig1 = sns.histplot(data=playerRatings, 
             x=selected_columns[0], 
             hue='Cluster', 
             bins=100, 
             kde=True, 
             stat="density", 
             palette='colorblind',
             multiple='layer',
             common_norm=False,
             fill=True,
             alpha=0.25)
plt.show()

fig, axes = plt.subplots(1,1, figsize=(10, 6), dpi=600)
fig2 = sns.histplot(data=playerRatings, 
             x=selected_columns[1], 
             hue='Cluster', 
             bins=100, 
             kde=True, 
             stat="density", 
             palette='colorblind',
             multiple='layer',
             common_norm=False,
             fill=True,
             alpha=0.25)
plt.show()

fig, axes = plt.subplots(1,1, figsize=(10, 6), dpi=600)
fig3 = sns.histplot(data=playerRatings, 
             x='ELO', 
             hue='Cluster', 
             bins=100, 
             kde=True, 
            # stat="density",
             palette='colorblind',
             multiple='layer',
             common_norm=False,
             fill=True,
             alpha=0.25)
plt.show()

ax1.get_figure().savefig(rf"{outputFolder}\exploration_diversityElbows.png", bbox_inches="tight")
fig1.get_figure().savefig(rf"{outputFolder}\exploration_gini3Cluster.png", bbox_inches="tight")
fig2.get_figure().savefig(rf"{outputFolder}\exploration_entropy3Cluster.png", bbox_inches="tight")
fig3.get_figure().savefig(rf"{outputFolder}\exploration_ELO3Cluster.png", bbox_inches="tight")



"""
###########################################################################################################################
Section 6
Clustering Tests - Diversity in player opening populations
###########################################################################################################################
"""

from scipy.stats import chi2_contingency
from scipy.cluster.hierarchy import linkage, dendrogram

# Create a contingency table of openings vs. players with counts
contingency_table = pd.crosstab(allRatings['Cluster'], allRatings[openingVariable])

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

plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table, annot=True, cmap='YlGnBu', cbar=True)
plt.title('Contingency Table Heatmap: Player Opening Choice vs. Rating')
plt.show()

"""
###########################################################################################################################
Section 6
Clustering Tests - Diversity in player opening populations
###########################################################################################################################
"""




modelTest_df = allRatings[[openingVariable, 'ELO', 'openingPlayer']]
df = modelTest_df
df=df.sort_values(by=['ELO'])
pivot_df = df.pivot_table(index=['openingPlayer', 'ELO'], columns=openingVariable, aggfunc='size', fill_value=0).reset_index()
sample_df =pivot_df
"""
sample_df,_  = train_test_split(pivot_df
                                , train_size=0.2
                                , random_state=123
                                #, stratify=pivot_df[openingVariable]
                                )
"""
sample_df=sample_df.drop(columns='openingPlayer')
mmscaler = MinMaxScaler(feature_range = (0,1), copy = False).fit(sample_df) 
sample_scaler = mmscaler.transform(sample_df)
sample_scaler = pd.DataFrame(sample_scaler, columns = sample_df.columns)



cluster_range = range(1, 11)  # Adjust as needed


silhouette_scores2 = []
silhouette_clusters2 = []

sse_scores2 = []
sse_clusters2 = []
# Iterate through different cluster numbers
for num_clusters in cluster_range:
    print(f"Start Cluster: {num_clusters}")
    kmeans = KMeans(n_clusters=num_clusters, random_state=123)
    cluster_labels = kmeans.fit_predict(sample_scaler)
    
    # Calculate the Sum of Squared Errors (SSE) for the elbow method
    sse = kmeans.inertia_
    sse_scores2.append(sse)
    sse_clusters2.append(num_clusters)
    
    if num_clusters>=2:
        # Calculate the silhouette score
        silhouette_avg = silhouette_score(sample_scaler, cluster_labels)
        silhouette_scores2.append(silhouette_avg)
        silhouette_clusters2.append(num_clusters)

# Plot WCSS (Elbow Method) on the left Y-axis
fig5, ax3 = plt.subplots(figsize=(10, 6), dpi=600)
ax3.plot(sse_clusters2, sse_scores2, marker='o', label='WCSS')
ax3.set_xlabel('Number of Clusters')
ax3.set_ylabel('WCSS')
#ax1.set_title('Elbow Method and Silhouette Score for Optimal Number of Clusters')

# Create a twin Y-axis on the right for Silhouette Score
ax4 = ax3.twinx()
ax4.plot(silhouette_clusters2, silhouette_scores2, marker='o', label='Silhouette Score', color='orange')
ax4.set_ylabel('Silhouette Score')

# Add legend
lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()
ax3.legend(lines2 + lines4, labels2 + labels4, loc="upper right")

# Show the plot
plt.grid(True)
plt.show()


# Perform K-means clustering with the chosen number of clusters
optimal_num_clusters = 3 # Adjust this based on your analysis
kmeans = KMeans(n_clusters=optimal_num_clusters)
sample_scaler['Cluster_openings'] = kmeans.fit_predict(sample_scaler)
sample_df['Cluster_openings'] = sample_scaler['Cluster_openings']

cluster_means = sample_scaler.groupby('Cluster_openings')['ELO'].mean().reset_index()
cluster_means_sorted = cluster_means.sort_values(by='ELO', ascending=True)  # Replace 'Rating' with the desired feature
cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(cluster_means_sorted['Cluster_openings'])}
sample_scaler['Cluster_openings'] = sample_scaler['Cluster_openings'].map(cluster_mapping)
pivot_df['Cluster_openings']=sample_scaler['Cluster_openings']
sample_df['Cluster_openings']=sample_scaler['Cluster_openings']


fig6, ax6 = plt.subplots(figsize=(10, 6), dpi=600)
fig6 = sns.histplot(data=pivot_df, 
             x='ELO', 
             hue='Cluster_openings', 
             bins=100, 
             kde=True, 
            # stat="density",
             palette='colorblind',
             multiple='stack',
             common_norm=False,
             fill=True,
             alpha=0.25)
plt.show()
fig7, ax7 = plt.subplots(figsize=(10, 6), dpi=600)
fig7 = sns.histplot(data=sample_df, 
             x='ELO', 
             hue='Cluster_openings', 
             bins=100, 
             kde=True, 
            # stat="density",
             palette='colorblind',
             multiple='layer',
             common_norm=False,
             fill=True,
             alpha=0.25)
plt.show()

fig5.get_figure().savefig(rf"{outputFolder}\exploration_cluster_openings_elbow.png", bbox_inches="tight")
fig6.get_figure().savefig(rf"{outputFolder}\exploration_cluster_openings_stack.png", bbox_inches="tight")
fig7.get_figure().savefig(rf"{outputFolder}\exploration_cluster_openings_layer.png", bbox_inches="tight")


allRatings = allRatings.merge(playerRatings[['openingPlayer', 'Cluster']], on='openingPlayer', how='left')

allRatings.to_csv(rf"{csvFolder}\allRatings.tsv", sep='\t')
playerRatings.to_csv(rf"{csvFolder}\playerRatings_CLUSTER.tsv", sep='\t')
openingAnalysis.to_csv(rf"{csvFolder}\openingAnalysis.tsv", sep='\t')
pivot_df[['ELO', 'openingPlayer', 'Cluster_openings']].to_csv(rf"{csvFolder}\playerOpenings_CLUSTER.tsv", sep='\t')










