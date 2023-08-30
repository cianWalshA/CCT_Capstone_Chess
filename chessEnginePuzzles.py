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
from pathlib import Path
from datetime import datetime 


stockfish_Path = Path(r"C:\Users\cianw\Chess Engines\Latest\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")
lc0_Path = Path(r"C:\Users\cianw\Chess Engines\Latest\lc0-v0.30.0-windows-gpu-nvidia-cuda\lc0.exe")
 
 
csvFolder = r"E:\ChessData"
pgnName = "lichess_db_standard_rated_2023-06_2000_5m"
pgnIn = Path(rf"{csvFolder}\{pgnName}.csv")
pgnIn_EnglineAnalysis = Path(rf"{csvFolder}\{pgnName}_engineApplied.tsv")


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


matePuzzles = lichessPuzzles[lichessPuzzles['Themes'].str.contains("mate", case=False)]
matePuzzles['mateInX'] = matePuzzles['Moves'].str.split().str.len()/2
matePuzzles['mateInX'].value_counts()
longMates = matePuzzles[matePuzzles['mateInX']>=5.0]


testDF= longMates.sample(n=10)

#matePuzzles['mateNum'] = matePuzzles.apply(lambda row: returnWord(row['Themes'], 'mate'), axis=1)


#lichessPuzzles['Stockfish_Analysis'] = lichessPuzzles.apply(lambda row: analyze_fen_moves(row['FEN'], row['Moves'], stockfish_engine), axis=1)
#lichessPuzzles['Lc0_Analysis'] = lichessPuzzles.apply(lambda row: analyze_fen_moves(row['FEN'], row['Moves'], lc0_engine), axis=1)


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
                     return pd.Series(info)
                elif info.get("time")>10:
                    return pd.Series(info)
                
    return pd.Series(info)
        

#longMates['StockfishMateDepth2'] = longMates.apply(lambda row: return_best_move_puzzles(row['FEN'], row['Moves'], stockfish_engine,20), axis=1)
#longMates['Lc0MateDepth2'] = longMates.apply(lambda row: return_best_move_puzzles(row['FEN'], row['Moves'], lc0_engine,20), axis=1)

longMates['stockfishInfo'] = longMates.apply(lambda row: return_best_move_puzzles_info(row['FEN'], row['Moves'], row['mateInX'], stockfish_engine, stockfish_options), axis=1)
longMates['lc0Info'] = longMates.apply(lambda row: return_best_move_puzzles_info(row['FEN'], row['Moves'], row['mateInX'], lc0_engine, lc0_options), axis=1)
"""
longMates['sf_time'] = longMates['stockfishInfo'].apply(lambda x: x.get('time'))
longMates['sf_depth'] = longMates['stockfishInfo'].apply(lambda x: x['depth'])
longMates['sf_seldepth'] = longMates['stockfishInfo'].apply(lambda x: x['seldepth'])
longMates['lc0_time'] = longMates['lc0Info'].apply(lambda x: x['time'])
longMates['lc0_depth'] = longMates['lc0Info'].apply(lambda x: x['depth'])
longMates['lc0_seldepth'] = longMates['lc0Info'].apply(lambda x: x['seldepth'])

longMates = pd.concat([longMates, pd.json_normalize(longMates["stockfishInfo"])], axis=1)

test=pd.json_normalize(longMates["stockfishInfo"])
test2= longMates["stockfishInfo"].apply(pd.Series).add_prefix("X_")
test3 = pd.json_normalize(longMates['stockfishInfo'])
"""
longMates.to_csv(r"C:\Users\cianw\Documents\dataAnalytics\projectFinal\Data\Chess\Lichess\puzzles\matePuzzleSolve2.csv")


testDF['stockfishInfo'] = testDF.apply(lambda row: return_best_move_puzzles_info(row['FEN'], row['Moves'], row['mateInX'], stockfish_engine, stockfish_options), axis=1)
testDF['lc0Info'] = testDF.apply(lambda row: return_best_move_puzzles_info(row['FEN'], row['Moves'], row['mateInX'], lc0_engine, lc0_options), axis=1)

newDF= testDF.apply(lambda row: return_best_move_puzzles_info(row['FEN'], row['Moves'], row['mateInX'], lc0_engine, lc0_options), axis=1)



stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_Path)
stockfish_engine.configure({'Clear Hash':True})

lc0_engine = chess.engine.SimpleEngine.popen_uci(lc0_Path)
lc0_engine.configure({'NNCacheSize':0})
fenTest = '5k2/p4p2/5Pp1/8/2P2pqr/2P4p/P4Q1P/4R2K b - - 0 46'
movesTest ='f4f3 f2c5 f8g8 e1e8 g8h7 e8h8 h7h8 c5f8 h8h7 f8g7'


firstMoveTest = movesTest.split()[0]
boardTest = chess.Board(fenTest)
boardTest.push_uci(firstMoveTest)


info = stockfish_engine.analyse(boardTest, limit=chess.engine.Limit(time=0), info=chess.engine.INFO_ALL)
while not info["score"].is_mate():
    info = stockfish_engine.analyse(boardTest, chess.engine.Limit())
    print(info["score"])
    
info2 = lc0_engine.analyse(boardTest, limit=chess.engine.Limit(time=0), info=chess.engine.INFO_ALL)
while not info2["score"].is_mate():
    info2 = lc0_engine.analyse(boardTest, chess.engine.Limit())
    print(info["score"])
        
    
start_time = time.time()
info = stockfish_engine.analyse(boardTest, chess.engine.Limit(mate=6))
print("--- %s seconds ---" % (time.time() - start_time))  
test=info["score"]
print(test)


start_time = time.time()
info2 = lc0_engine.analyse(boardTest, chess.engine.Limit())
print("--- %s seconds ---" % (time.time() - start_time))  
test2=info2["score"]
print(test2)

info = stockfish_engine.analyse(boardTest, limit=chess.engine.Limit(time=0), info=chess.engine.INFO_ALL)
with stockfish_engine.analysis(boardTest) as analysis:
    for info in analysis:
        print(info.get("score"), info.get("pv"))
        print(type(info.get("score")))
        # Arbitrary stop condition.
        if info.get("depth", 0) > 10:
             break
with stockfish_engine.analysis(boardTest) as analysis:
    for info in analysis:
        print(info.get("score"), info.get("pv"))
        print(type(info.get("score")))
        # Arbitrary stop condition.
        if info.get("score"):
            if info.get("score").is_mate() == True :
                 break
with lc0_engine.analysis(boardTest) as analysis:
    for info in analysis:
        print(info.get("score"), info.get("pv"))
        print(type(info.get("score")))
        # Arbitrary stop condition.
        if info.get("score"):
            if info.get("score").relative in (chess.engine.Mate(5), chess.engine.Mate(-5)):
                 break
        
boardTest.uci(info2["pv"][0])
best_move = boardTest.uci(info.move)
best_move_algebraic = boardTest.uci(best_move)

re.findall(r"(\w+f\w+)",movesTest)


substring2 = 'amp'
test_string = 'this is an example of the text that I have'

print("matches for substring 1:",re.findall(r"(\w+he text th\w+)", test_string))
print("matches for substring 2:",re.findall(r"(\w+ha\w+)",test_string))
"""
	FEN
2357530	5k2/p4p2/5Pp1/8/2P2pqr/2P4p/P4Q1P/4R2K b - - 0 46

	Moves
2357530	f4f3 f2c5 f8g8 e1e8 g8h7 e8h8 h7h8 c5f8 h8h7 f8g7
"""

with stockfish_engine.analysis(boardTest) as analysis:
     for info in analysis:
         print(info.get("score"), info.get("pv"))

         # Arbitrary stop condition.
         if info.get("seldepth", 0) > 20:
             break

stockfish_engine.quit()
