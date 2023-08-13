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


stockfish_Path = Path(r"C:\Users\cianw\Chess Engines\stockfish_15.1_win_x64_avx2\stockfish-windows-2022-x86-64-avx2.exe")
lc0_Path = Path(r"C:\Users\cianw\Chess Engines\lc0-v0.29.0-windows-gpu-nvidia-cuda\lc0.exe")
 
 
csvFolder = r"E:\ChessData"
pgnName = "lichess_db_standard_rated_2023-06_2000_5m"
pgnIn = Path(rf"{csvFolder}\{pgnName}.csv")
pgnIn_EnglineAnalysis = Path(rf"{csvFolder}\{pgnName}_engineApplied.tsv")


lichessPuzzles_Path = Path(r"C:\Users\cianw\Documents\dataAnalytics\projectFinal\Data\Chess\Lichess\puzzles\lichess_db_puzzle.csv")
lichessPuzzles = pd.read_csv(lichessPuzzles_Path)

stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_Path)
lc0_engine = chess.engine.SimpleEngine.popen_uci(lc0_Path)

def analyze_fen_moves(fen, moves, engine,):
    board = chess.Board(fen)
    firstMove = moves.split()[0]
    otherMoves = moves.split(' ', 1)[1]
    board.push_uci(firstMove)
    for move_str in otherMoves.split():
        move = chess.Move.from_uci(move_str)
        board.push(move)
    info = engine.analyse(board, limit=chess.engine(time=0.1))

    return info

def returnWord(s, substr):
    pattern = rf'\b\w*{substr}\w*\b'
    match = re.search(pattern, s, re.IGNORECASE)
    return match.group() if match else None


matePuzzles = lichessPuzzles[lichessPuzzles['Themes'].str.contains("mate", case=False)]
longMates = matePuzzles[matePuzzles['Themes'].str.contains("mateIn5", case=False)]

test= longMates.sample(n=10)

matePuzzles['mateNum'] = matePuzzles.apply(lambda row: returnWord(row['Themes'], 'mate'), axis=1)


lichessPuzzles['Stockfish_Analysis'] = lichessPuzzles.apply(lambda row: analyze_fen_moves(row['FEN'], row['Moves'], stockfish_engine), axis=1)
lichessPuzzles['Lc0_Analysis'] = lichessPuzzles.apply(lambda row: analyze_fen_moves(row['FEN'], row['Moves'], lc0_engine), axis=1)




def return_best_move_puzzles(fen, moves, engine, inputDepth):
    for i in range(1,inputDepth+1):
        print(i)
        board = chess.Board(fen)
        firstMove = moves.split()[0]   
        secondMove = moves.split()[1]   
        board.push_uci(firstMove)
        info = engine.analyse(board, limit=chess.engine.Limit(depth=i, time = 10))
        chosenMove=board.uci(info["pv"][0])
        if board.turn ==True:
            turn='White'
        else:
            turn='Black'
        evaluation = info["score"][0]
        usedDepth = info["depth"]
        multiPV = info["multipv"]
        if chosenMove == secondMove:
            return usedDepth, multiPV, chosenMove, evaluation, turn
        elif usedDepth==inputDepth:
            return usedDepth, multiPV, chosenMove, evaluation, turn
        else:
            pass
        

longMates['StockfishMateDepth2'] = longMates.apply(lambda row: return_best_move_puzzles(row['FEN'], row['Moves'], stockfish_engine,20), axis=1)
longMates['Lc0MateDepth2'] = longMates.apply(lambda row: return_best_move_puzzles(row['FEN'], row['Moves'], lc0_engine,20), axis=1)
test[['sfDepth','sfPV','sfMove', 'sfEval', 'sfTurn']] = test.apply(lambda row: return_best_move_puzzles(row['FEN'], row['Moves'], stockfish_engine,1), axis=1,result_type='expand')
test[['lc0Depth','lc0PV','lc0Move', 'lc0Eval','lc0Turn']] = test.apply(lambda row: return_best_move_puzzles(row['FEN'], row['Moves'], lc0_engine,1), axis=1, result_type='expand')
longMates.to_csv(r"C:\Users\cianw\Documents\dataAnalytics\projectFinal\Data\Chess\Lichess\puzzles\matePuzzleSolve.csv")
    

fenTest = '5k2/p4p2/5Pp1/8/2P2pqr/2P4p/P4Q1P/4R2K b - - 0 46'
movesTest ='f4f3 f2c5 f8g8 e1e8 g8h7 e8h8 h7h8 c5f8 h8h7 f8g7'
firstMoveTest = movesTest.split()[0]
boardTest = chess.Board(fenTest)
boardTest.push_uci(firstMoveTest)
info = stockfish_engine.analyse(boardTest, limit=chess.engine.Limit(depth=1, time= 0.0001))
test=info.score
test2=info["score"]
boardTest.uci(info["pv"][0])
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
