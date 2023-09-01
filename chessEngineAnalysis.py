

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
pgnOut = Path(rf"{csvFolder}\{pgnName}_output.tsv")

lichessData = pd.read_csv(pgnIn, nrows=100)
lichessData['UTC_dateTime'] = pd.to_datetime(lichessData['UTCDate'] + ' ' + lichessData['UTCTime'])
lichessData.describe()

#moveTest = "1. e4 e5 2. Nf3 Nc6 3. Bc4 d6 4. h3 Nf6 5. Nc3 Be6 6. Be2 Qd7 7. O-O O-O-O 8. Re1 h6 9. d3 Bxh3 10. gxh3 Qxh3 11. Bf1 Qg4+ 12. Bg2 h5 13. Nh2 Qe6 14. Be3 g5 15. d4 exd4 16. Bxd4 h4 17. Nd5 Bg7 18. Nxf6 Bxf6 19. Bxf6 Qxf6 20. Bh3+ Kb8 21. c3 Rdg8 22. Ng4 Qe7 23. b4 Ne5 24. Re3 a6 25. Kg2 Rg6 26. Qd4 Rf8 27. Nxe5 dxe5 28. Qc4 g4 29. Rd1 gxh3+ 30. Kxh3 Qg5 31. Qf1 f5 32. exf5 Qxf5+ 33. Kh2 Qf4+ 34. Kh1 Qxe3 35. fxe3 Rxf1+ 36. Rxf1 Rg3 37. Re1 e4 38. Kh2 Kc8 39. Re2 Kd7 40. c4 Ke6 41. a4 Kf5 42. Kh1 Kg4 43. Kh2 Kf3 44. Re1 Kf2 45. Rd1 Kxe3 46. Rd7 Kf2 47. Rxc7 e3 48. Rf7+ Ke1 49. Rxb7 e2 50. Re7 Kf2 51. b5 axb5 52. cxb5 Re3 53. Rf7+ Ke1 54. b6 Kd2 55. Rd7+ Kc3 56. Rc7+ Kb4 57. Rc1 e1=Q 58. Rxe1 Rxe1"

stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_Path)
stockfish_options = {'Clear Hash':True}
lc0_engine = chess.engine.SimpleEngine.popen_uci(lc0_Path)
lc0_options = {'NNCacheSize':0}

def evaluateGame(games, loadedEngine, engineOptions,gameNumber):
    gameNumber+=1
    gameMoves = chess.pgn.read_game(io.StringIO(games['Moves']))
    gameMoves.headers

    board = gameMoves.board()
    evalList1 = []
    depthList1 = []
    evalList2 = []
    depthList2 = []
    evalDiff =[]
    depthDiff =[]
    for move in gameMoves.mainline_moves():
        loadedEngine.configure(engineOptions)
        info1 = loadedEngine.analyse(board, limit=chess.engine.Limit(time=0.01, depth=20))
        score1 = info1['score'].white()
        evalList1.append(score1)
        depthList1.append(info1['depth'])
        #evalDiff.append(score2-score1)
        #depthDiff.append(info2['depth']-info1['depth'])
        """
        #check for mate and assign high value either way, maths says infinity but might not be suitable for analysis later
        if score1.is_mate():
            numericScore1 = float(10000) if score1.mate() > 0 else float(-10000)
        else:
            numericScore1 = score1.cp / 100.0
        if engine2!=None:
            info2 = engine2.analyse(board, limit=chess.engine.Limit(time=0.01, depth=20))
            score2 = info2['score']
            #check for mate and assign high value either way, maths says infinity but might not be suitable for analysis later
            if score2.is_mate():
                numericScore2 = float(10000) if score2.mate() > 0 else float(-10000)
            else:
                numericScore2 = score2.cp / 100.0
        
        evalList1.append(numericScore1)
        depthList1.append(info1['depth'])
        evalList2.append(numericScore2)
        depthList2.append(info2['depth'])
        evalDiff.append(numericScore2-numericScore1)
        depthDiff.append(info2['depth']-info1['depth'])
        """
        #print(f"Score: {info['score']},\tScore (white): " + f"{info['score'].white()},\tDepth: {info['depth']}")
        board.push(move)
    """
    return evalList1, depthList1, evalList2, depthList2, evalDiff, depthDiff
    """
    if gameNumber%10 == 0:
        print(rf"{gameNumber} - Games Processed")
    return evalList1, depthList1
"""
start_time = time.time()
lichessData[['stockfish_eval','stockfish_depth', 
             'Lc0_eval','Lc0_depth', 
             'evalDiff','depthDiff']] = lichessData.apply(evaluateGame,
                                                          engine1=engineStockfish,
                                                          engine2=engineLc0, 
                                                          axis=1, 
                                                          result_type='expand')
"""

start_time = time.time()
lichessData[['stockfish_eval','stockfish_depth']] = lichessData.apply(evaluateGame,
    loadedEngine=stockfish_engine,
    engineOptions = stockfish_options,
    gameNumber=0,
    axis=1, 
    result_type='expand')
print("--- %s seconds ---" % (time.time() - start_time))      

start_time = time.time()
lichessData[['lc0_eval','lc0_depth']] = lichessData.apply(evaluateGame,
    loadedEngine = lc0_engine,
    engineOptions = lc0_options,
    gameNumber=0,
    axis=1, 
    result_type='expand')           
print("--- %s seconds ---" % (time.time() - start_time))   #                                        




#lichessData.to_csv(pgnOut, sep="\t")

lichessDate_read = pd.read_csv(pgnOut, sep="\t")



















