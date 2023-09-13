

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
import plotly.express as px
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime 

stockfish_Path = Path(r"C:\Users\cianw\Chess Engines\Latest\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")
lc0_Path = Path(r"C:\Users\cianw\Chess Engines\Latest\lc0-v0.30.0-windows-gpu-nvidia-cuda\lc0.exe")
 
pgnFolder = r"E:\ChessData"
csvFolder = r"E:\ChessData"
pgnName = "lichess_db_standard_rated_2023-06_2000_5m"
pgnIn = Path(rf"{csvFolder}\{pgnName}.csv")
pgnOut = Path(rf"{csvFolder}\{pgnName}_output.tsv")

lichessData = pd.read_csv(pgnIn)
lichessData['UTC_dateTime'] = pd.to_datetime(lichessData['UTCDate'] + ' ' + lichessData['UTCTime'])
lichessData.describe()

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

#moveTest = "1. e4 e5 2. Nf3 Nc6 3. Bc4 d6 4. h3 Nf6 5. Nc3 Be6 6. Be2 Qd7 7. O-O O-O-O 8. Re1 h6 9. d3 Bxh3 10. gxh3 Qxh3 11. Bf1 Qg4+ 12. Bg2 h5 13. Nh2 Qe6 14. Be3 g5 15. d4 exd4 16. Bxd4 h4 17. Nd5 Bg7 18. Nxf6 Bxf6 19. Bxf6 Qxf6 20. Bh3+ Kb8 21. c3 Rdg8 22. Ng4 Qe7 23. b4 Ne5 24. Re3 a6 25. Kg2 Rg6 26. Qd4 Rf8 27. Nxe5 dxe5 28. Qc4 g4 29. Rd1 gxh3+ 30. Kxh3 Qg5 31. Qf1 f5 32. exf5 Qxf5+ 33. Kh2 Qf4+ 34. Kh1 Qxe3 35. fxe3 Rxf1+ 36. Rxf1 Rg3 37. Re1 e4 38. Kh2 Kc8 39. Re2 Kd7 40. c4 Ke6 41. a4 Kf5 42. Kh1 Kg4 43. Kh2 Kf3 44. Re1 Kf2 45. Rd1 Kxe3 46. Rd7 Kf2 47. Rxc7 e3 48. Rf7+ Ke1 49. Rxb7 e2 50. Re7 Kf2 51. b5 axb5 52. cxb5 Re3 53. Rf7+ Ke1 54. b6 Kd2 55. Rd7+ Kc3 56. Rc7+ Kb4 57. Rc1 e1=Q 58. Rxe1 Rxe1"

stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_Path)
stockfish_options = {'Clear Hash':True, 'Threads': 4}
stockfish_engine.configure(stockfish_options)
#lc0_engine = chess.engine.SimpleEngine.popen_uci(lc0_Path)
#lc0_options = {'NNCacheSize':0}

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

def get_final_fen(pgn):
    bongCloudPosition = io.StringIO(pgn)
    game = chess.pgn.read_game(bongCloudPosition)
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    return board.fen()

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
        elif (moveCount-games['halfMoveCount'])%10==0 and (moveCount-games['halfMoveCount'])>=0:
            info1 = loadedEngine.analyse(board, limit=chess.engine.Limit(time=0.1), info=chess.engine.INFO_ALL)
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


"""
Dataset Creation
"""

openings = pd.concat([openings_a, openings_b, openings_c, openings_d, openings_e])
del openings_a, openings_b, openings_c, openings_d, openings_e

openings['moveNumbers'] = openings['pgn'].apply(lambda x: extract_nth_words(x, 1, 3))
openings['whiteMoves'] = openings['pgn'].apply(lambda x: extract_nth_words(x, 2, 3))
openings['blackMoves'] = openings['pgn'].apply(lambda x: extract_nth_words(x, 3, 3))
openings['moveCount'] = openings['moveNumbers'].str.split().str.len()
openings['halfMoveCount'] = openings['whiteMoves'].str.split().str.len() + openings['blackMoves'].str.split().str.len()  
openings['white_black'] = openings['halfMoveCount'].apply(lambda x: 'black' if x % 2 == 0 else 'white') 
# Sort the DataFrame by columns 'a' and 'b'
openings = openings.sort_values(by=['name', 'halfMoveCount'])
openings = openings.drop_duplicates(subset='name', keep='first')

lichessData = lichessData.merge(openings, left_on='Opening', right_on='name', how='left')
lichessData = lichessData.dropna(subset=['name']) 
lichessData['openingPlayer'] = np.where(lichessData['white_black'] == 'black', lichessData['Black'], lichessData['White'])


"""
Dataset Exploration
"""

lichessData = lichessData.sort_values(by='Opening', ascending=False)

openingsPlayed = lichessData.groupby('Opening').size().reset_index(name='Count')
uniquePlayers = lichessData.groupby('Opening')['openingPlayer'].nunique().reset_index().rename(columns={'openingPlayer': 'countOpeningPlayers'})
openingAnalysis = openingsPlayed.merge(uniquePlayers, on='Opening')
openingAnalysis['openingPlayerDiversity'] =(openingAnalysis['Count']-openingAnalysis['countOpeningPlayers'])/openingAnalysis['Count']


uniquePlayerOpenings = lichessData.groupby('openingPlayer')['Opening'].nunique().reset_index().rename(columns={'Opening': 'countPlayerOpening'})
uniquePlayerGames = lichessData.groupby('openingPlayer').size().reset_index(name='countGames')
playerAnalysis = uniquePlayerOpenings.merge(uniquePlayerGames, on='openingPlayer')
playerAnalysis['openingsUsedDivesity'] =(playerAnalysis['countGames']-playerAnalysis['countPlayerOpening'])/playerAnalysis['countGames']
playerAnalysisFiltered =playerAnalysis[(playerAnalysis['countPlayerOpening']>5) & (playerAnalysis['openingsUsedDivesity']>0.1)]


lichessData = lichessData.merge(openingAnalysis, on='Opening', how='left')
lichessData = lichessData.merge(playerAnalysis, on='openingPlayer', how='left')
lichessDataRatioFilter = lichessData[(lichessData['countOpeningPlayers']>100) & (lichessData['openingPlayerDiversity']>0.1)]


category_counts = lichessData['Opening'].value_counts().reset_index().rename(columns={'index': 'Opening', 'Opening': 'counts'})
percentiles = [99, 95, 90, 85, 80, 75,70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 1]
countPercentiles = np.percentile(category_counts['counts'], percentiles)
percentilesCategory = pd.DataFrame({'Percentile': percentiles, 'Value': countPercentiles})

category_counts2 = lichessDataRatioFilter['Opening'].value_counts().reset_index().rename(columns={'index': 'Opening', 'Opening': 'counts'})
percentiles = [99, 95, 90, 85, 80, 75,70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 1]
countPercentiles2 = np.percentile(category_counts2['counts'], percentiles)
percentilesCategory2 = pd.DataFrame({'Percentile': percentiles, 'Value': countPercentiles2})

category_counts = category_counts.sort_values(by='Count', ascending=False)



lichessDataSummary = lichessData['Opening'].value_counts().reset_index()
lichessData = lichessData.merge(lichessDataSummary, left_on='Opening', right_on='index', how='left')



                                           
"""
Application of Engine Analysis
"""

chunk_size = 100  # Adjust this based on your memory constraints
# Define your data processing function here
def process_data(chunk):
    chunk_out = pd.DataFrame()
    chunk_out[['SF_eval','SF_depth','SF_seldepth']] = chunk.apply(evaluateGame,
        loadedEngine=stockfish_engine,
        engineOptions = stockfish_options,
        axis=1, 
        result_type='expand')
    return pd.concat([chunk, chunk_out], axis=1)


linesProcessed = 0
dataFrameSize = len(lichessData)
printThreshold = dataFrameSize/1000
start_time = time.time()
processed_df = pd.DataFrame()
for start_idx in range(0, len(lichessData), chunk_size):
    startTime = time.time()
    end_idx = min(start_idx + chunk_size, len(lichessData))
    chunk = lichessData.iloc[start_idx:end_idx]
    
    try:
        # Process the chunk and add new columns
        processed_chunk = process_data(chunk)
        processed_df = pd.concat([processed_df, processed_chunk], ignore_index=True)
    except Exception as e:
        print(f"Error occurred: {e}")
    print(time.time()-startTime)

# Save the final processed DataFrame to a file
#processed_df.to_csv(pgnOut, sep="\t")

#readin1 = pd.read_csv("E:\ChessData\lichess_db_standard_rated_2023-06_2000_160000_rows.tsv", sep = "\t")
#readin2 = pd.read_csv("E:\ChessData\lichess_db_standard_rated_2023-06_2000_remainder.tsv", sep = "\t")

#stacked_df = pd.concat([readin1, readin2])

#stacked_df.to_csv("E:\ChessData\lichess_db_standard_rated_2023-06_2000_1m_row_analysed.tsv", sep="\t")














