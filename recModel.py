# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 10:14:38 2023

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

from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale

import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.models import Model
from keras.layers import Reshape
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import random

"""
###########################################################################################################################
Section 0
Imports
###########################################################################################################################
"""
stockfish_Path = Path(r"C:\Users\cianw\Chess Engines\Latest\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")
lc0_Path = Path(r"C:\Users\cianw\Chess Engines\Latest\lc0-v0.30.0-windows-gpu-nvidia-cuda\lc0.exe")
 
pgnFolder = r"E:\ChessData"
csvFolder = r"E:\ChessData\explorationOutputs"
csvAllRatingFolder = r"E:\ChessData\explorationOutputs"
outputFolder = r"C:\Users\cianw\Documents\dataAnalytics\projectFinal\figureOutputs"

inDataGames_Name = "allRatings"
inDataOpening_Name = 'openingAnalysis'
inDataPlayerOpenings_Name = 'playerOpenings'
inDataPlayerOpeningsClusters_Name = 'playerOpenings_CLUSTER'
inDataPlayerRatingsClusters_Name = 'playerRatings_CLUSTER'
engineAnalysis_Name = 'allRatings_output_20230918'

games_PATH = Path(rf"{csvFolder}\{inDataGames_Name}.tsv")
openingAnalysis_PATH = Path(rf"{csvFolder}\{inDataOpening_Name}.tsv")
playerOpenings_PATH = Path(rf"{csvFolder}\{inDataPlayerOpenings_Name}.tsv")
playerOpeningsClusters_PATH = Path(rf"{csvFolder}\{inDataPlayerOpeningsClusters_Name}.tsv")
playerRatingsClusters_PATH = Path(rf"{csvFolder}\{inDataPlayerRatingsClusters_Name}.tsv")
engineAnalysis_path = Path(rf"{csvFolder}\{engineAnalysis_Name}.tsv")

#games = pd.read_csv(games_PATH, sep='\t')
#games['UTC_dateTime'] = pd.to_datetime(games['UTCDate'] + ' ' + games['UTCTime'])
#games.describe()

games_exploration = pd.read_csv(games_PATH, sep='\t')
games_exploration['UTC_dateTime'] = pd.to_datetime(games_exploration['UTCDate'] + ' ' + games_exploration['UTCTime'])
games_exploration.describe()

openingAnalysis = pd.read_csv(openingAnalysis_PATH, sep='\t')
openingAnalysis.describe()

playerOpenings = pd.read_csv(playerOpenings_PATH, sep='\t')
playerOpenings.describe()

playerOpeningsClusters = pd.read_csv(playerOpeningsClusters_PATH, sep='\t')
playerOpeningsClusters.describe()

playerRatingsClusters = pd.read_csv(playerRatingsClusters_PATH, sep='\t')
playerRatingsClusters.describe()

engineAnalysis = pd.read_csv(engineAnalysis_path, sep='\t')
engineAnalysis.describe()

"""
###########################################################################################################################
Section 1
Functions
###########################################################################################################################
"""
#Function to Extract Every Nth Word of a string delimted by spaces starting at position M, up to N*O words.
def extract_nth_words(text, M, N, O=None):
    words = text.strip().split(' ')
    if O is None:
        endIndex = len(words)
    else: 
        endIndex = min(M-1+N*O, len(words))
    result = [words[i] for i in range(M - 1, endIndex, N)]
    return ' '.join(result)

def remove_moves(row, moves):
    # Split the whiteMoves string into a list of moves
    moves = row[moves].split()
    # Determine the number of moves to remove based on 'removeMoves' value
    num_moves_to_remove = row['removeMoves']
    # Remove the specified number of moves
    updated_moves = moves[num_moves_to_remove:]
    # Join the updated moves back into a single string
    return ' '.join(updated_moves)


#Function to return every ith element of a list that was saved as a string
def get_ith_element(lst, i):
    res = lst.strip('][').split(', ')
    if len(res) >= i+1:
        return res[i]
    else:
        return None

#Function to return every ith element of a list that was saved as a string
def get_ith_element(lst, i):
    res = lst.strip('][').split(', ')
    if len(res) >= i+1:
        return res[i]
    else:
        return None

def get_final_fen(pgn):
    bongCloudPosition = io.StringIO(pgn)
    game = chess.pgn.read_game(bongCloudPosition)
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    return board.fen()

def extract_features(pgn, X, Y):    # Initialize a dictionary to store the features
    features = {
        'pawn_moves': 0,
        'center_pawns': 0,
        'flank_pawns':0,
        'piece_moves': 0,
        'developed_pieces': 0,
        'center_pieces': 0,
        'minor_moves': 0,
        'queen_moves': 0,
        'queen_retreat': 0,
        'king_moves': 0,
        'king_forward': 0,
        'minor_retreat':0,
        'castles': 0,
        'checks': 0,
        'captures': 0
    }

    moves = pgn.split()

    # Extract moves within the specified range [X, Y]
    moves = moves[X:(Y+X)]
    movesPlayed = len(moves)

    for move in moves:
        # Ignore move numbers
        if '.' not in move:
            # Count pawn moves (e.g., e4, d5)
            if re.match('^[a-h][1-8]$', move):
                features['pawn_moves'] += (1/movesPlayed)
                # Count center pawns
                if re.match('^[de][45]$', move):
                    features['center_pawns'] += (1/movesPlayed)
                if re.match('^[ah][1-8]$', move):
                    features['flank_pawns'] += (1/movesPlayed)
            # Count piece moves (e.g., Nf3, Bb5)
            if re.match('^[NBRQK][a-h][1-8]$', move):
                features['piece_moves'] += (1/movesPlayed)
                # Count developed pieces
                if re.match('^[NBR][a-h][2-8]$', move):
                    features['developed_pieces'] += (1/movesPlayed)
                #Count Minor Retreat
                if re.match('^[NBR][a-h][1]$', move):
                    features['minor_retreat'] += (1/movesPlayed)
                # Count center pieces
                if re.match('^[NBRQK][de][45]$', move):
                    features['center_pieces'] += (1/movesPlayed)
                # Count Minor Piece Moves
                if re.match('^[NBR][a-h][1-8]$', move):
                    features['minor_moves'] += (1/movesPlayed)
                # Count Queen Moves
                if re.match('^[Q][a-h][2-8]$', move):
                    features['queen_moves'] += (1/movesPlayed)
                # Count Queen retreat
                if re.match('^[Q][a-h][1]$', move):
                    features['queen_retreat'] += (1/movesPlayed)
                # Count King Moves
                if re.match('^[K][a-h][1]$', move):
                    features['king_moves'] += (1/movesPlayed)    
                # Count King Forward
                if re.match('^[K][a-h][2-8]$', move):
                    features['king_forward'] += (1/movesPlayed)   
            # Count castling moves (O-O or O-O-O)
            if 'O-O' in move:
                features['castles'] += (1/movesPlayed)
            # Count checks (+)
            if '+' in move:
                features['checks'] += (1/movesPlayed)
            # Count captures (x)
            if 'x' in move:
                features['captures'] += (1/movesPlayed)
    return features


"""
###########################################################################################################################
Section 1
Final Pre-Processing
###########################################################################################################################
"""
openingVariable = 'Opening'

openingsList = games_exploration['Opening'].unique()
playersList = games_exploration['openingPlayer'].unique()
keepCols = ['openingPlayer',openingVariable, 'Moves', 'halfMoveCount', 'openingPlayerRating', 'white_black', 'ELO']
games_exploration=games_exploration[keepCols]


#Define Clusters for Model Use, Remove overlap of rating based on medians
playerOpeningsClusters = playerOpeningsClusters[playerOpeningsClusters['openingPlayer'].isin(playersList)]
playerOpeningsClusters = playerOpeningsClusters.sort_values(['ELO'])


openingAnalysis = openingAnalysis[openingAnalysis['Opening'].isin(openingsList)]
openingAnalysis['successRatio'] = openingAnalysis['whiteWin']/openingAnalysis['timesPlayed']

rds = games_exploration.merge(playerOpeningsClusters[['openingPlayer', 'Cluster_openings']], on='openingPlayer', how='left' )

#Get white and black moves and number of moves in opening
rds['whiteMoves'] = rds['Moves'].apply(lambda text: ' '.join(text.split()[1::3]))
rds['blackMoves'] = rds['Moves'].apply(lambda text: ' '.join(text.split()[2::3]))

rds['openingMovesWhite'] = ((rds['halfMoveCount']+1)/2).astype(int)
rds['openingMovesBlack'] = ((rds['halfMoveCount']-1)/2).astype(int)

#Get behavior data and join
whiteBehavior20 = pd.json_normalize(rds.apply(lambda row: extract_features(row['whiteMoves'], row['openingMovesWhite'], 20), axis=1)).add_suffix("_20").fillna(0)
behaviorColumns = whiteBehavior20.columns.tolist()
rds = rds.join(whiteBehavior20)

matrixCols = behaviorColumns.copy()  
matrixCols.append('openingWinProbability')

#Create player/opening aggregated dataset
rds_agg = rds.groupby(['openingPlayer', openingVariable])[behaviorColumns].mean().reset_index()
rds_agg = rds_agg.merge(playerOpenings[['openingPlayer', openingVariable, 'openingWinProbability']], on=['openingPlayer', openingVariable], how='left')
rds_agg = rds_agg.merge(playerOpeningsClusters[['openingPlayer', 'Cluster_openings']], on=['openingPlayer'], how='left').fillna(0)

#Create player aggregated dataset
rds_player_agg = rds.groupby(['openingPlayer'])[behaviorColumns].mean().reset_index()
rds_player_agg = rds_player_agg.merge(playerOpeningsClusters[['openingPlayer', 'Cluster_openings']], on=['openingPlayer'], how='left').fillna(0)



#Create scaled datasets for models
scaler = MinMaxScaler()
rds_player_agg_scaled=pd.DataFrame()
rds_player_agg_scaled = scaler.fit_transform(rds_player_agg[behaviorColumns])
rds_player_agg_scaled = pd.DataFrame(rds_player_agg_scaled, columns = behaviorColumns)
rds_player_agg_scaled = rds_player_agg_scaled.join(rds_player_agg[['openingPlayer', 'Cluster_openings']])

scaler = MinMaxScaler()
rds_agg_scaled=pd.DataFrame()
rds_agg_scaled = scaler.fit_transform(rds_agg[matrixCols])
rds_agg_scaled = pd.DataFrame(rds_agg_scaled, columns = matrixCols)
rds_agg_scaled = rds_agg_scaled.join(rds_agg[['openingPlayer', 'Opening', 'Cluster_openings']])


player_behaviour_agg = rds.groupby(['openingPlayer'])[behaviorColumns].median().reset_index()
opening_behaviour_agg = rds.groupby(['openingPlayer'])[behaviorColumns].median().reset_index()

playerOpeningCombinations = playerOpenings[['openingPlayer', 'Opening']]

"""
distinctPlayers = rds_agg['openingPlayer'].unique()
random.seed(123)
distinctPlayerSample = random.sample(distinctPlayers.tolist(),10000)

distinctOpenings = rds_agg['Opening'].unique()
playerOpeningCombinations = rds_agg[['openingPlayer', 'Opening']]

rds_sample=rds[rds['openingPlayer'].isin(distinctPlayerSample)].reset_index(drop=True)
rds_agg_sample = rds_agg[rds_agg['openingPlayer'].isin(distinctPlayerSample)].reset_index(drop=True)
rds_player_agg_sample = rds_player_agg[rds_player_agg['openingPlayer'].isin(distinctPlayerSample)].reset_index(drop=True)
"""

"""
###########################################################################################################################
Section 2
Examine Engine Evaluation
###########################################################################################################################
"""


engineAnalysis = engineAnalysis.join(pd.DataFrame(engineAnalysis['SF_eval'].apply(ast.literal_eval).values.tolist()).add_prefix('eval')).drop(columns={'SF_eval'})
engineAnalysis = engineAnalysis.join(pd.DataFrame(engineAnalysis['SF_seldepth'].apply(ast.literal_eval).values.tolist()).add_prefix('seldepth')).drop(columns={'SF_seldepth'})


selectedCols = [col for col in engineAnalysis.columns if 'eval' in col]
engineSummary = engineAnalysis.groupby('Opening')[selectedCols].describe()
engineSummary.columns = [' '.join(col).strip() for col in engineSummary.columns.values]
medianCols = [col for col in engineSummary.columns if '50%' in col]
engineSummary = engineSummary[medianCols].reset_index()

scaler = MinMaxScaler()
engineSummary_scaled = scaler.fit_transform(engineSummary[medianCols])
engineSummary_scaled = pd.DataFrame(engineSummary_scaled, columns = medianCols)
engineSummary_scaled = engineSummary_scaled.join(engineSummary['Opening'])

engineSummary = engineSummary.merge(openingAnalysis[['Opening','successRatio']], on='Opening', how='left')

analysisCols = medianCols.copy()
analysisCols.append('successRatio')
sns.pairplot(engineSummary[analysisCols])
plt.show()

# Create a correlation matrix
correlation_matrix = engineSummary[analysisCols].corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

usefulCols=['Opening', 'eval0 50%','eval1 50%'] 

openingAnalysis = openingAnalysis.merge(engineSummary_scaled[usefulCols], on='Opening', how='left')
openingAnalysis[[ 'eval0 50%','eval1 50%']]=openingAnalysis[[ 'eval0 50%','eval1 50%']].fillna(0)





"""
###########################################################################################################################
Section 2
Multi-Feature Behavior Model - Regression
###########################################################################################################################
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor




X_train, X_test, y_train, y_test = train_test_split(rds_agg_scaled[behaviorColumns], rds_agg_scaled['openingWinProbability'], test_size=0.33, random_state=123)

X_train.columns = behaviorColumns
X_test.columns = behaviorColumns

models = [
    ('Linear Regression', LinearRegression()),
    ('Lasso Regression', Lasso()),
    ('Random Forest Regression', RandomForestRegressor()),
    ('Decision Tree Regression', DecisionTreeRegressor()),
    ('XGBoost Regressor', XGBRegressor()),
]


# Train and evaluate each regression model
modelTestResults = []

for model_name, model in models:
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    modelTestResults.append((model_name, mse, r2))


pd.DataFrame(modelTestResults).to_csv(rf"{csvFolder}\modelTestResults.tsv", sep='\t')


"""
###########################################################################################################################
Section 2
Multi-Feature Behavior Model - Euclidean Distance
###########################################################################################################################
"""
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from tqdm import tqdm

def knn_recommender_multi(df, players, opening_var, player_var, ranking_var, interaction_table, playerOpenings, nrecs):
    print("Start Cluster Process")
    interaction_matrix = csr_matrix(interaction_table.values)
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(interaction_matrix)
    
    recommendations_list = []
    #countPlayers = 0
    numPlayers = len(players)
    for player in tqdm(players):
        #countPlayers+=1
        #playerPct = ((100*countPlayers)/numPlayers)
        #print(countPlayers)
        #if (playerPct)%1 == 0:
       #     print(f"{playerPct}% Complete")
        for k in range(5,30,1):
            distances, indices = model_knn.kneighbors(interaction_table.loc[player,:].values.reshape(1,-1), n_neighbors = k)

            players = []
            distance = []
            
            for i in range(0, len(distances.flatten())):
                if i != 0:
                    players.append(interaction_table.index[indices.flatten()[i]])
                    distance.append(distances.flatten()[i])    
        
            m=pd.Series(players,name=player_var)
            d=pd.Series(distance,name='distance')
            recommend = pd.concat([m,d], axis=1)
            recommend = recommend.sort_values('distance',ascending=False)
            
            playerOpeningsUsed = playerOpenings[playerOpenings[player_var]==player][opening_var].tolist()
            
            similarPlayers = recommend[player_var].tolist()
            similarPlayersOpenings =  playerOpenings[playerOpenings[player_var].isin(similarPlayers)][[player_var, opening_var]]
            similarPlayersOpenings = similarPlayersOpenings[~similarPlayersOpenings[opening_var].isin(playerOpeningsUsed)].drop_duplicates(opening_var)
            
            if len(similarPlayersOpenings[opening_var]) <nrecs:
                continue 
            else:
                similarPlayersOpenings = similarPlayersOpenings.merge(openingAnalysis, on=opening_var, how='left')
                similarPlayersOpenings =similarPlayersOpenings.sort_values(ranking_var, ascending=False)
                recommendedOpenings = similarPlayersOpenings.nlargest(nrecs, ranking_var)[opening_var].tolist()
                recommendations_list.append([player, recommendedOpenings])
                break
            
    recommendations_df = pd.DataFrame(recommendations_list, columns=[player_var, 'Recommendations'])
    return recommendations_df

"""
Combinations
Cluster, Behavior, winProbability, eval0 50%, eval1 50%
"""
#Eval based
po_df = rds_agg_scaled
po_players = po_df['openingPlayer'].unique().tolist()
po_winProb_inter_table = po_df.pivot_table(index = ["openingPlayer"], columns='Opening', values = 'openingWinProbability').fillna(0)

rec_po_sr_df = knn_recommender_multi(po_winProb_inter_table
                               , po_players
                               , 'Opening'
                               , 'openingPlayer'
                               , 'successRatio'
                               , po_winProb_inter_table
                               , playerOpeningCombinations
                               ,10
                               )
rec_po_sr_df.to_csv(rf"{csvFolder}\rec_po_sr_df.tsv", sep='\t')

rec_eval0_sr_df = knn_recommender_multi(po_winProb_inter_table
                               , po_players
                               , 'Opening'
                               , 'openingPlayer'
                               , 'eval0 50%'
                               , po_winProb_inter_table
                               , playerOpeningCombinations
                               ,10
                               )
rec_eval0_sr_df.to_csv(rf"{csvFolder}\rec_eval0_sr_df.tsv", sep='\t')

rec_eval1_sr_df = knn_recommender_multi(po_winProb_inter_table
                               , po_players
                               , 'Opening'
                               , 'openingPlayer'
                               , 'eval1 50%'
                               , po_winProb_inter_table
                               , playerOpeningCombinations
                               ,10
                               )
rec_eval1_sr_df.to_csv(rf"{csvFolder}\rec_eval1_sr_df.tsv", sep='\t')

#Behaviour Based
p_df = rds_player_agg_scaled
p_players = p_df['openingPlayer'].unique().tolist()
p_beh_inter_table = p_df.pivot_table(index = ["openingPlayer"], values = behaviorColumns).fillna(0)
rec_beh_sr_df = knn_recommender_multi(p_df
                               , p_players
                               , 'Opening'
                               , 'openingPlayer'
                               , 'successRatio'
                               , p_beh_inter_table
                               , playerOpeningCombinations
                               ,10
                               )
rec_beh_sr_df.to_csv(rf"{csvFolder}\rec_beh_sr_df.tsv", sep='\t')




#Cluster Behavior
clusters =[0, 1, 2]


recommendation_cluster_behavior_df = pd.DataFrame()
for i in clusters:
    
    cluster_df = rds_player_agg_scaled[rds_player_agg_scaled['Cluster_openings']==i].reset_index(drop=True)
    cluster_players = cluster_df['openingPlayer'].unique().tolist()
    inter_table_clust_beh = cluster_df.pivot_table(index = ["openingPlayer"], values = behaviorColumns).fillna(0)
    rec_df = knn_recommender_multi(cluster_df
                                   , cluster_players
                                   , 'Opening'
                                   , 'openingPlayer'
                                   , 'successRatio'
                                   , inter_table_clust_beh
                                   , playerOpeningCombinations
                                   ,10
                                   )
    rec_df['cluster'] = i
    recommendation_cluster_behavior_df = pd.concat(recommendation_cluster_behavior_df,rec_df)

recommendation_cluster_behavior_df.to_csv(rf"{csvFolder}\rec_clust_behav.tsv", sep='\t')

recommendation_cluster_sr_df = pd.DataFrame()
for i in clusters:
    
    cluster_df1 = rds_agg_scaled[rds_agg_scaled['Cluster_openings']==i].reset_index(drop=True)
    cluster_players1 = cluster_df1['openingPlayer'].unique().tolist()
    inter_table_clust_sr = cluster_df1.pivot_table(index = ["openingPlayer"], columns='Opening', values = 'openingWinProbability').fillna(0)
    rec_sr_df = knn_recommender_multi(cluster_df1
                                   , cluster_players1
                                   , 'Opening'
                                   , 'openingPlayer'
                                   , 'successRatio'
                                   , inter_table_clust_sr
                                   , playerOpeningCombinations
                                   ,10
                                   )
    rec_sr_df['cluster'] = i
    recommendation_cluster_sr_df = pd.concat(recommendation_cluster_sr_df,rec_sr_df)
recommendation_cluster_sr_df.to_csv(rf"{csvFolder}\rec_clust_sr.tsv", sep='\t')


