# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 20:26:40 2023

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
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from datetime import datetime 


stockfish_Path = Path(r"C:\Users\cianw\Chess Engines\Latest\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")
lc0_Path = Path(r"C:\Users\cianw\Chess Engines\Latest\lc0-v0.30.0-windows-gpu-nvidia-cuda\lc0.exe")
 
 
csvFolder = r"E:\ChessData"
pgnName = "lichess_db_standard_rated_2023-06_2000_5m"
pgnIn = Path(rf"{csvFolder}\{pgnName}.csv")
pgnIn_EnglineAnalysis = Path(rf"{csvFolder}\{pgnName}_engineApplied.tsv")

outputFolder = r"C:\Users\cianw\Documents\dataAnalytics\projectFinal\Outputs"

lichessPuzzles_Path = Path(r"C:\Users\cianw\Documents\dataAnalytics\projectFinal\Data\Chess\Lichess\puzzles\lichess_db_puzzle.csv")
lichessPuzzles = pd.read_csv(lichessPuzzles_Path)

stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_Path)
stockfish_options = {'Clear Hash':True}
lc0_engine = chess.engine.SimpleEngine.popen_uci(lc0_Path)
lc0_options = {'NNCacheSize':0}


def returnWord(s, substr):
    pattern = rf'\b\w*{substr}\w*\b'
    match = re.search(pattern, s, re.IGNORECASE)
    return match.group() if match else None

def return_best_move_puzzles_info(fen, moves, mateInX, loadedEngine, engineOptions):
    outputList = []
    #print(i)
    board = chess.Board(fen)
    firstMove = moves.split()[0]   
    secondMove = moves.split()[1]   
    board.push_uci(firstMove)
    loadedEngine.configure(engineOptions)
    #Declares a dictionary of "info" where it can be looped through while the mate has not been found
    info = loadedEngine.analyse(board, limit=chess.engine.Limit(time=0), info=chess.engine.INFO_ALL)
    with loadedEngine.analysis(board) as analysis:
        for info in analysis:
            if info.get("score"):
                if info.get("score").relative in (chess.engine.Mate(mateInX), chess.engine.Mate(mateInX)):
                    break
                elif info.get("time")>10:
                    break
    return info

def chess_position_converter(value):
    return chess.Board(value)      

def check_for_mate(df, variable):
    contains_mate = df[variable].str.contains('Mate').sum()
    does_not_contain_mate = len(df) - contains_mate
    print(f"{variable} - Number of strings containing 'Mate': {contains_mate}")
    print(f"{variable} - Number of strings not containing 'Mate': {does_not_contain_mate}")
    
    
matePuzzles = lichessPuzzles[lichessPuzzles['Themes'].str.contains("mate", case=False)]
matePuzzles['mateInX'] = matePuzzles['Moves'].str.split().str.len()/2
matePuzzles['mateInX'].value_counts()
longMates = matePuzzles[matePuzzles['mateInX']>=5.0]


testDF= longMates.sample(n=10)


longMates['stockfishInfo'] = longMates.apply(lambda row: return_best_move_puzzles_info(row['FEN'], row['Moves'], row['mateInX'], stockfish_engine, stockfish_options), axis=1)
longMates['lc0Info'] = longMates.apply(lambda row: return_best_move_puzzles_info(row['FEN'], row['Moves'], row['mateInX'], lc0_engine, lc0_options), axis=1)
longMates = pd.concat([longMates,longMates['stockfishInfo'].apply(pd.Series).add_prefix('SF_')], axis=1)
longMates = pd.concat([longMates,longMates['lc0Info'].apply(pd.Series).add_prefix('LC0_')], axis=1)
longMates.to_csv(r"C:\Users\cianw\Documents\dataAnalytics\projectFinal\Data\Chess\Lichess\puzzles\matePuzzlesSolvedExp.csv")

#Save and load in again, removes chess object type for both SF_score and LC0_score, but this is not an issue for analysis
solved = pd.read_csv(r"C:\Users\cianw\Documents\dataAnalytics\projectFinal\Data\Chess\Lichess\puzzles\matePuzzlesSolvedExp.csv")

solved_Adj = solved[solved['mateInX']<=8]
check_for_mate(solved_Adj, 'SF_score')
check_for_mate(solved_Adj, 'LC0_score')

solved_Adj = solved_Adj[(solved_Adj['SF_score'].str.contains('Mate')) & (solved_Adj['LC0_score'].str.contains('Mate'))]

uniqueVals = pd.Series.unique(solved['mateInX'])

#Test for normality in distributions:
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

#Test for normality as Time is Continouse
#Nodes and seldepth are discrete therefore non-normal
SF_timeNormal = normal_test_subgroups(solved_Adj, 'SF_time', 'mateInX', 0.05)
LC0_timeNormal = normal_test_subgroups(solved_Adj, 'LC0_time', 'mateInX', 0.05)
            

def mwu_test(df, dataCol1, dataCol2, subgroup, direction , alpha):
     results=[]
     uniqueVals = pd.Series.unique(df[subgroup])
     for value in uniqueVals:
         subgroup_df = df[df[subgroup]==int(value)]
         
         #Apply Shapiro-Wilk Normality Test
         summary1= subgroup_df[dataCol1].describe(include='all').loc['50%']
         summary2= subgroup_df[dataCol2].describe(include='all').loc['50%']

         tStat, pValue = stats.mannwhitneyu(x = subgroup_df[dataCol1], y = subgroup_df[dataCol2], alternative = direction, method='auto')
         # Determine if the subgroup follows a normal distribution
         testResult = pValue < alpha
         
         results.append((value, tStat, pValue, testResult, summary1,summary2))
         print()
         print(rf"Variable: {dataCol1}&{dataCol2} - Subgroup: {value} - test: {tStat} - p-Value: {pValue} - Result: {testResult}")
     return pd.DataFrame(results, columns = [subgroup, 'testStatistic', 'pValue', 'testResult','summaryValue1', 'summaryValue2' ])  

solved_Adj_rename = solved_Adj.rename(columns={'SF_seldepth':'Stockfish SelDepth',
                                        'LC0_seldepth':'LeelaChessZero SelDepth',
                                        'SF_time':'Stockfish Time',
                                        'LC0_time':'LeelaChessZero Time',
                                        'SF_nodes':'Stockfish Nodes',
                                        'LC0_nodes':'LeelaChessZero Nodes',
                                        'mateInX':'Mate In X'
                                        })

sns.histplot(data=solved_Adj_rename, x='Stockfish Time', bins= 100)
plt.show()
sns.histplot(data=solved_Adj_rename, x='LeelaChessZero Time')
plt.show()
    
seldepthDistTest = mwu_test(solved_Adj_rename, 'Stockfish SelDepth', 'LeelaChessZero SelDepth', 'Mate In X','less', 0.05)
timeDistTest = mwu_test(solved_Adj_rename, 'Stockfish Time', 'LeelaChessZero Time', 'Mate In X','less', 0.05)
nodesDistTest = mwu_test(solved_Adj_rename, 'Stockfish Nodes', 'LeelaChessZero Nodes', 'Mate In X','greater', 0.05)



sns.set(style="whitegrid")
plt.figure(figsize=(10, 6), dpi=600)
seldepthMelt = pd.melt(solved_Adj_rename,id_vars = 'Mate In X', value_vars=['Stockfish SelDepth','LeelaChessZero SelDepth' ])
seldepthMelt = seldepthMelt.rename(columns={'value':'SelDepth', 'variable':'Engine'})
fig1 = sns.boxplot(data=seldepthMelt, x='Engine', y= 'SelDepth', hue= 'Mate In X', palette='Set3')
plt.legend(title="Mate in X", loc="upper left", bbox_to_anchor=(0.275, -0.1), ncol=4)
y_ticks = np.arange(0, seldepthMelt["SelDepth"].max() + 1, 5)
plt.yticks(y_ticks)
plt.show()

plt.figure(figsize=(10, 6), dpi=600)
timeMelt = pd.melt(solved_Adj_rename,id_vars = 'Mate In X', value_vars=['Stockfish Time','LeelaChessZero Time' ])
timeMelt = timeMelt.rename(columns={'value':'Time (ms)', 'variable':'Engine'})
fig2 = sns.boxplot(data=timeMelt, x='Engine', y= 'Time (ms)', hue= 'Mate In X', palette='Set3')
plt.legend(title="Mate in X", loc="upper left", bbox_to_anchor=(0.275, -0.1), ncol=4)
plt.ylabel('Log(Time (ms))')
plt.yscale('log')
plt.show()

plt.figure(figsize=(10, 6), dpi=600)
nodesMelt = pd.melt(solved_Adj_rename,id_vars = 'Mate In X', value_vars=['Stockfish Nodes','LeelaChessZero Nodes' ])
nodesMelt = nodesMelt.rename(columns={'value':'Nodes', 'variable':'Engine'})
fig3 = sns.boxplot(data=nodesMelt, x='Engine', y= 'Nodes', hue= 'Mate In X', palette='Set3')
plt.legend(title="Mate in X", loc="upper left", bbox_to_anchor=(0.275, -0.1), ncol=4)
plt.ylabel('Log(Nodes)')
plt.yscale('log')
plt.show()

fig1.get_figure().savefig(rf"{outputFolder}\SelDepth.png", bbox_inches="tight")
fig2.get_figure().savefig(rf"{outputFolder}\Time.png", bbox_inches="tight")
fig3.get_figure().savefig(rf"{outputFolder}\Nodes.png", bbox_inches="tight")

