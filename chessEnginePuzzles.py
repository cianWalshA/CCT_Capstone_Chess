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
lichessPuzzles = pd.read_csv(lichessPuzzles_Path,nrows=10)

lichess = chess.engine.SimpleEngine.popen_uci(stockfish_Path)
lc0 = chess.engine.SimpleEngine.popen_uci(lc0_Path)


lichessPuzzles['Stockfish_Analysis'] = lichessPuzzles.apply(lambda row: analyze_fen_moves(row['FEN'], row['Moves'], stockfish), axis=1)
lichessPuzzles['Lc0_Analysis'] = lichessPuzzles.apply(lambda row: analyze_fen_moves(row['FEN'], row['Moves'], lc0), axis=1)

stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_Path)
lc0 = chess.engine.SimpleEngine.popen_uci(lc0_Path)

def analyze_fen_moves(fen, moves, engine):
    board = chess.Board(fen)
    if moves:
        first_move = moves.split()[0]
        move = chess.Move.from_uci(first_move)
        san_move = board.san(move)
        board.push_san(san_move)
    for move_str in moves.split():
        move = chess.Move.from_uci(move_str)
        board.push(move)

    info = engine.analyse(board, limit=chess.engine(time=0.1))

    return info

lichessPuzzles['Stockfish_Analysis'] = lichessPuzzles.apply(lambda row: analyze_fen_moves(row['FEN'], row['Moves'], stockfish), axis=1)
lichessPuzzles['Lc0_Analysis'] = lichessPuzzles.apply(lambda row: analyze_fen_moves(row['FEN'], row['Moves'], lc0), axis=1)


fenTest= 'r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24'

boardTest = chess.Board(fenTest)
info = stockfish.analyse(boardTest, limit=chess.engine.Limit(time=0.1))
best_move = info.get("pv", [])[0]
best_move_algebraic = boardTest.san(best_move)










