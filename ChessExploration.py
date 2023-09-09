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
from pathlib import Path
from datetime import datetime 


stockfish_Path = Path(r"C:\Users\cianw\Chess Engines\Latest\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")
lc0_Path = Path(r"C:\Users\cianw\Chess Engines\Latest\lc0-v0.30.0-windows-gpu-nvidia-cuda\lc0.exe")
 
pgnFolder = r"E:\ChessData"
csvFolder = r"E:\ChessData"
pgnName = "lichess_db_standard_rated_2023-06_2000_1m_row_analysed"
pgnIn_EnglineAnalysis = Path(rf"{csvFolder}\{pgnName}.tsv")


lichessData = pd.read_csv(pgnIn_EnglineAnalysis, sep='\t', nrows=100000)
lichessData['UTC_dateTime'] = pd.to_datetime(lichessData['UTCDate'] + ' ' + lichessData['UTCTime'])
lichessData.describe()

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
Exploration of Engine Analysed Sub-Sample
"""



"""
SECTION - DEF
Chess Puzzles Testing for Engines
"""







