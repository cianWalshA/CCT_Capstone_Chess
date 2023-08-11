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

matePuzzles['mateNum'] = matePuzzles.apply(lambda row: returnWord(row['Themes'], 'mate'), axis=1)


lichessPuzzles['Stockfish_Analysis'] = lichessPuzzles.apply(lambda row: analyze_fen_moves(row['FEN'], row['Moves'], stockfish_engine), axis=1)
lichessPuzzles['Lc0_Analysis'] = lichessPuzzles.apply(lambda row: analyze_fen_moves(row['FEN'], row['Moves'], lc0_engine), axis=1)


fenTest = '7k/pp2q1p1/1n2p1Bp/1b3pN1/1n1P3P/8/PP1K1PP1/1Q5R b - - 0 24'
movesTest ='h6g5 h4g5 h8g8 h1h8 g8h8 b1h1 h8g8 h1h7 g8f8 h7h8'
firstMoveTest = movesTest.split()[0]

def return_best_move_puzzles(fen, moves, engine, inputDepth):
    for i in range(1,inputDepth+1):
        board = chess.Board(fen)
        firstMove = moves.split()[0]   
        secondMove = moves.split()[1]   
        board.push_uci(firstMove)
        info = engine.play(board, limit=chess.engine.Limit(depth=i, time = 0.1))
        if board.uci(info.move) == secondMove:
            return i
        else:
            pass
        

longMates['StockfishMateDepth'] = longMates.apply(lambda row: return_best_move_puzzles(row['FEN'], row['Moves'], stockfish_engine,10), axis=1)
longMates['Lc0MateDepth'] = longMates.apply(lambda row: return_best_move_puzzles(row['FEN'], row['Moves'], lc0_engine,10), axis=1)

longMates.to_csv(r"C:\Users\cianw\Documents\dataAnalytics\projectFinal\Data\Chess\Lichess\puzzles\matePuzzleSolve.csv")
    
boardTest = chess.Board(fenTest)
boardTest.push_uci(firstMoveTest)
info = lc0_engine.play(boardTest, limit=chess.engine.Limit(depth=1))
test=info.score
best_move = info.move
best_move_algebraic = boardTest.uci(best_move)

re.findall(r"(\w+f\w+)",movesTest)


substring2 = 'amp'
test_string = 'this is an example of the text that I have'

print("matches for substring 1:",re.findall(r"(\w+he text th\w+)", test_string))
print("matches for substring 2:",re.findall(r"(\w+ha\w+)",test_string))



