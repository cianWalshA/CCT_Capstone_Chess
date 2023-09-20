

import sys
import io
import os
import ast
import time
import csv
import chess
import chess.pgn
import stockfish
import re
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime 


stockfish_Path = Path(r"C:\Users\cianw\Chess Engines\Latest\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")
lc0_Path = Path(r"C:\Users\cianw\Chess Engines\Latest\lc0-v0.30.0-windows-gpu-nvidia-cuda\lc0.exe")
 
outputName = "allRatings"
pgnFolder = r"E:\ChessData"
csvFolder = r"E:\ChessData\explorationOutputs"
pgnName = "allRatings"
pgnIn = Path(rf"{csvFolder}\{pgnName}.tsv")
pgnOut = Path(rf"{csvFolder}\{pgnName}_output_20230918.tsv")
pgnOut_iter = Path(rf"{csvFolder}\{pgnName}_output_20230918_iter.tsv")

lichessData = pd.read_csv(pgnIn, sep = "\t")
lichessData['UTC_dateTime'] = pd.to_datetime(lichessData['UTCDate'] + ' ' + lichessData['UTCTime'])
lichessData.describe()

openingVariable = 'Opening'

#moveTest = "1. e4 e5 2. Nf3 Nc6 3. Bc4 d6 4. h3 Nf6 5. Nc3 Be6 6. Be2 Qd7 7. O-O O-O-O 8. Re1 h6 9. d3 Bxh3 10. gxh3 Qxh3 11. Bf1 Qg4+ 12. Bg2 h5 13. Nh2 Qe6 14. Be3 g5 15. d4 exd4 16. Bxd4 h4 17. Nd5 Bg7 18. Nxf6 Bxf6 19. Bxf6 Qxf6 20. Bh3+ Kb8 21. c3 Rdg8 22. Ng4 Qe7 23. b4 Ne5 24. Re3 a6 25. Kg2 Rg6 26. Qd4 Rf8 27. Nxe5 dxe5 28. Qc4 g4 29. Rd1 gxh3+ 30. Kxh3 Qg5 31. Qf1 f5 32. exf5 Qxf5+ 33. Kh2 Qf4+ 34. Kh1 Qxe3 35. fxe3 Rxf1+ 36. Rxf1 Rg3 37. Re1 e4 38. Kh2 Kc8 39. Re2 Kd7 40. c4 Ke6 41. a4 Kf5 42. Kh1 Kg4 43. Kh2 Kf3 44. Re1 Kf2 45. Rd1 Kxe3 46. Rd7 Kf2 47. Rxc7 e3 48. Rf7+ Ke1 49. Rxb7 e2 50. Re7 Kf2 51. b5 axb5 52. cxb5 Re3 53. Rf7+ Ke1 54. b6 Kd2 55. Rd7+ Kc3 56. Rc7+ Kb4 57. Rc1 e1=Q 58. Rxe1 Rxe1"

stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_Path)
stockfish_options = {'Clear Hash':True, 'Threads': 4}
stockfish_engine.configure(stockfish_options)
#lc0_engine = chess.engine.SimpleEngine.popen_uci(lc0_Path)
#lc0_options = {'NNCacheSize':0}



def evaluateGame(games, loadedEngine, engineOptions):
    global linesProcessed, dataFrameSize, printThreshold, start_time
    
    gameMoves = chess.pgn.read_game(io.StringIO(games['Moves']))
    gameMoves.headers

    board = gameMoves.board()
    evalList1 = []
    depthList1 = []
    seldepthList1 = []
    loadedEngine.configure(engineOptions)
    moveCount=0
    for move in gameMoves.mainline_moves():
        board.push(move)
        moveCount+=1
        if moveCount<games['halfMoveCount'] :
            pass
        elif ((moveCount-games['halfMoveCount'])/10)==5:
            break
        elif ((moveCount-games['halfMoveCount']))%10==0 and (moveCount-games['halfMoveCount'])>=0:
            info1 = loadedEngine.analyse(board, limit=chess.engine.Limit(time=1), info=chess.engine.INFO_ALL)
            score1 = info1['score'].white().score()
            evalList1.append(score1)
            depthList1.append(info1['depth'])
            if info1.get("seldepth", 0):
                seldepthList1.append(info1.get("seldepth", 0))
            else:
                seldepthList1.append(None)
    linesProcessed += 1
    if linesProcessed%1000 == 0:
        print(linesProcessed)
        print((time.time() - start_time))
    return evalList1, depthList1, seldepthList1

def process_data(chunk):
    chunk_out = pd.DataFrame()
    chunk_out[['SF_eval','SF_depth','SF_seldepth']] = chunk.apply(  evaluateGame,
                                                                    loadedEngine=stockfish_engine,
                                                                    engineOptions = stockfish_options,
                                                                    axis=1, 
                                                                    result_type='expand')
    return pd.concat([chunk, chunk_out], axis=1)
def csvCommit(outFile, dictToWrite, csvHeaders):
    file_exists = os.path.isfile(outFile)
    with open(outFile, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=csvHeaders)
        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
        for row in dictToWrite:
            writer.writerow(row)
"""
Dataset Creation
"""

lichessData['winLossDraw'] = lichessData['Result'].str.split('-').str[0]
df = lichessData

from sklearn.model_selection import train_test_split


sample_df,_ = train_test_split(df
                               , train_size=0.01
                               , random_state=123
                               , stratify=df[[openingVariable, 'whiteWin']])
                                     
"""
Application of Engine Analysis
"""

chunk_size = 500  # Adjust this based on your memory constraints
# Define your data processing function here



linesProcessed = 0
dataFrameSize = len(df)
printThreshold = dataFrameSize/1000
start_time = time.time()
analysis_df = sample_df
processed_df = pd.DataFrame()

for start_idx in range(0, len(analysis_df), chunk_size):
    startTime = time.time()
    end_idx = start_idx + chunk_size
    chunk = analysis_df.iloc[start_idx:end_idx]
    
    try:
        # Process the chunk and add new columns
        processed_chunk = process_data(chunk)
        processed_df = pd.concat([processed_df, processed_chunk], ignore_index=True)
    except Exception as e:
        print(f"Error occurred: {e}")
    
    if os.path.exists(pgnOut_iter):
        processed_chunk.to_csv(pgnOut_iter, sep="\t", mode='a', header=False)
    else:
        processed_chunk.to_csv(pgnOut_iter, sep="\t", mode='w')
        
    
    print(time.time()-startTime)
    print(f"{start_idx} Completed")

# Save the final processed DataFrame to a file
processed_df.to_csv(pgnOut, sep="\t")











