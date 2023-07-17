# -*- coding: utf-8 -*-
"""
Created on Sat May  6 11:19:23 2023

@author: cianw
"""
import numpy as np
import pandas as pd
from pathlib import Path
 
from chester.timecontrol import TimeControl
from chester.tournament import play_tournament
leelaPath = Path(rf"C:\Users\cianw\Chess Engines\lc0-v0.29.0-windows-gpu-nvidia-cuda\lc0.exe")
stockfishPath = Path(rf"C:\Users\cianw\Chess Engines\stockfish_15.1_win_x64_avx2\stockfish-windows-2022-x86-64-avx2.exe")

openingBookPath = Path(rf"C:\Users\cianw\Chess Engines\OpeningBooks\BIN\Perfect2023.bin")

    
import chess
import chess.engine
import pandas as pd

def play_game(engine1, engine2, depth, time):
    board = chess.Board()
    game = []
    while not board.is_game_over():
        current_engine = engine1 if board.turn else engine2
        result = current_engine.play(board, chess.engine.Limit(time=time, depth=depth))
        board.push(result.move)
        game.append(board.fen())

    result_mapping = {'1-0': 'Engine 1 wins', '0-1': 'Engine 2 wins', '1/2-1/2': 'Draw'}
    result = result_mapping.get(board.result(), 'Unknown')
    print("Finito")

    return game, result

def collect_data(engine1, engine2, depths, times):
    data = []
    for depth in depths:
        for time in times:
            game, result = play_game(engine1, engine2, depth, time)
            data.append((depth, time, game, result))
    df = pd.DataFrame(data, columns=['Depth', 'Time', 'Game', 'Result'])
    return df

# Initialize chess engines
engine1 = chess.engine.SimpleEngine.popen_uci(leelaPath)
engine2 = chess.engine.SimpleEngine.popen_uci(stockfishPath)

# Define parameter ranges
depths = [1]
times = [0.1,0.1,0.1,0.1, 1,1,1,1,10,10,10,10, 25, 25,25, 25]

# Collect data
df = collect_data(engine1, engine2, depths, times)

# Close engines
engine1.quit()
engine2.quit()

# Print the dataframe
print(df)

