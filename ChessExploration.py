# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 20:40:31 2023

@author: cianw
"""
import sys
import io
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
 
pgnFolder = r"C:\Users\cianw\Documents\dataAnalytics\projectFinal\Data\Chess\Lichess"
csvFolder = r"C:\Users\cianw\Documents\dataAnalytics\projectFinal\Data\Chess\Lichess_CSV"
pgnName = "lichess_db_standard_rated_2013-01"
pgnIn = Path(rf"{csvFolder}\{pgnName}.csv")
pgnIn_EnglineAnalysis = Path(rf"{csvFolder}\{pgnName}_output.tsv")

lichessData = pd.read_csv(pgnIn)
lichessData['UTC_dateTime'] = pd.to_datetime(lichessData['UTCDate'] + ' ' + lichessData['UTCTime'])
lichessData.describe()

lichessData_EA = pd.read_csv(pgnIn_EnglineAnalysis, sep='\t')
lichessData_EA['UTC_dateTime'] = pd.to_datetime(lichessData_EA['UTCDate'] + ' ' + lichessData_EA['UTCTime'])
lichessData_EA.describe()


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
    
    
    
    

"""
Section XYZ - Feature Extraction from Complete Set
"""


lichessData['moveNumbers'] = lichessData['Moves'].apply(lambda x: extract_nth_words(x, 1, 3))
lichessData['moveCount'] = lichessData['moveNumbers'].str.split().str.len()
lichessData['whiteMoves'] = lichessData['Moves'].apply(lambda x: extract_nth_words(x, 2, 3))
lichessData['blackMoves'] = lichessData['Moves'].apply(lambda x: extract_nth_words(x, 3, 3))
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
lichessData_EA['stockfish_eval_test'] = lichessData_EA['stockfish_eval'].apply(get_ith_element, i=5)

lichessData_EA['sex'].value_counts()


"""
SECTION - DEF
Chess Puzzles Testing for Engines
"""







