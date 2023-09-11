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
from pathlib import Path
from datetime import datetime 


stockfish_Path = Path(r"C:\Users\cianw\Chess Engines\Latest\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")
lc0_Path = Path(r"C:\Users\cianw\Chess Engines\Latest\lc0-v0.30.0-windows-gpu-nvidia-cuda\lc0.exe")
 
pgnFolder = r"E:\ChessData"
csvFolder = r"E:\ChessData"
csvAllRatingFolder = r"E:\ChessData\newOutputs"

pgnName = "lichess_db_standard_rated_2023-06_2000_1m_row_analysed"
pgnIn_EnglineAnalysis = Path(rf"{csvFolder}\{pgnName}.tsv")

pgnNameAllRating = "lichess_db_standard_rated_2023-06__allRatings"
pgnIn_AllRatings = Path(rf"{csvAllRatingFolder}\{pgnNameAllRating}.csv")


lichessData = pd.read_csv(pgnIn_EnglineAnalysis, sep='\t', nrows=100000)
lichessData['UTC_dateTime'] = pd.to_datetime(lichessData['UTCDate'] + ' ' + lichessData['UTCTime'])
lichessData.describe()


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



lichessData = lichessData.join(pd.DataFrame(lichessData['SF_eval'].apply(ast.literal_eval).values.tolist()).add_prefix('eval')).drop(columns={'SF_eval'})
lichessData = lichessData.join(pd.DataFrame(lichessData['SF_seldepth'].apply(ast.literal_eval).values.tolist()).add_prefix('seldepth')).drop(columns={'SF_seldepth'})

lichessData = lichessData.dropna(subset='eval4')

selectedCols = [col for col in lichessData.columns if col.startswith('eval')]
lichessSummary = lichessData.groupby('Opening_x')[selectedCols].describe()
lichessSummary.columns = [' '.join(col).strip() for col in lichessSummary.columns.values]





"""
Section XYZ - Feature Extraction from Complete Set
"""
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


#Count and plot how many games based on move length

#Count and plot how many games based on each opening



#Where move count >= 5, >=10 return 10th and 20th values of rating respectively?

"""
SECTION - ABC
Exploration of rating distribution
"""

openings['moveNumbers'] = openings['pgn'].apply(lambda x: extract_nth_words(x, 1, 3))
openings['whiteMoves'] = openings['pgn'].apply(lambda x: extract_nth_words(x, 2, 3))
openings['blackMoves'] = openings['pgn'].apply(lambda x: extract_nth_words(x, 3, 3))
openings['moveCount'] = openings['moveNumbers'].str.split().str.len()
openings['halfMoveCount'] = openings['whiteMoves'].str.split().str.len() + openings['blackMoves'].str.split().str.len()  
openings['white_black'] = openings['halfMoveCount'].apply(lambda x: 'black' if x % 2 == 0 else 'white') 
# Sort the DataFrame by columns 'a' and 'b'
openings = openings.sort_values(by=['name', 'halfMoveCount'])
openings = openings.drop_duplicates(subset='name', keep='first')

allRatings = allRatings.merge(openings, left_on='Opening', right_on='name', how='left')
allRatings = allRatings.dropna(subset=['name']) 
allRatings['openingPlayer'] = np.where(allRatings['white_black'] == 'black', allRatings['Black'], allRatings['White'])
allRatings['openingPlayerRating'] = np.where(allRatings['white_black'] == 'black', allRatings['Black'], allRatings['White'])


openingsPlayed = allRatings.groupby('Opening').size().reset_index(name='Count')
uniquePlayers = allRatings.groupby('Opening')['openingPlayer'].nunique().reset_index().rename(columns={'openingPlayer': 'countOpeningPlayers'})
openingAnalysis = openingsPlayed.merge(uniquePlayers, on='Opening')
openingAnalysis['openingPlayerDiversity'] =(openingAnalysis['Count']-openingAnalysis['countOpeningPlayers'])/openingAnalysis['Count']


uniquePlayerOpenings = allRatings.groupby('openingPlayer')['Opening'].nunique().reset_index().rename(columns={'Opening': 'countPlayerOpening'})
uniquePlayerGames = allRatings.groupby('openingPlayer').size().reset_index(name='countGames')
playerAnalysis = uniquePlayerOpenings.merge(uniquePlayerGames, on='openingPlayer')
playerAnalysis['openingsUsedDivesity'] =(playerAnalysis['countGames']-playerAnalysis['countPlayerOpening'])/playerAnalysis['countGames']


allRatings = allRatings.merge(openingAnalysis, on='Opening', how='left')
allRatings = allRatings.merge(playerAnalysis, on='openingPlayer', how='left')
allRatingsRatioFilter = allRatings[(allRatings['countOpeningPlayers']>100) & (allRatings['openingPlayerDiversity']>0.1)]

playerRatings = allRatings.groupby('openingPlayer').size().reset_index(name='countGames')

whiteElo = allRatings[['White','WhiteElo']].rename(columns={'White':'Player', 'WhiteElo':'ELO'})
blackElo = allRatings[['Black','BlackElo']].rename(columns={'Black':'Player', 'BlackElo':'ELO'})

playerRatingsBoth = pd.concat([whiteElo,blackElo])
playerRatings = playerRatingsBoth.groupby('Player')['ELO'].mean().reset_index(name='ELO')
playerRatings=playerRatings.merge(uniquePlayerOpenings, left_on='Player', right_on='openingPlayer', how='left')
playerRatings = playerRatings.dropna(subset=['openingPlayer'])

allRatings = allRatings.merge(playerRatings, left_on='openingPlayer', right_on='Player', how='left')

playerRatings['ELO'] = pd.cut(playerRatings['ELO'], bins=6, labels=False)


# Create a histogram of player ratings
plt.figure(figsize=(8, 4))
plt.hist(playerRatings['ELO'], bins=300, edgecolor='k', alpha=0.75)
plt.xlabel('Player Rating')
plt.ylabel('Number of Players')
plt.title('Distribution of Player Ratings')
plt.grid(True)
plt.show()

# Create a scatter plot of Rating vs. UniqueOpenings
plt.figure(figsize=(8, 6))
plt.scatter(playerRatings['ELO'], playerRatings['countPlayerOpening'], alpha=0.75)
plt.xlabel('Player Rating')
plt.ylabel('Number of Unique Openings Played')
plt.title('Player Rating vs. Unique Openings Played')
plt.grid(True)
plt.show()

"""
SECTION - ABC
Exploration of Engine Analysed Sub-Sample
"""



"""
SECTION - DEF
Chess Puzzles Testing for Engines
"""







