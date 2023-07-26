

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

stockfish_Path = Path(rf"C:\Users\cianw\Chess Engines\stockfish_15.1_win_x64_avx2\stockfish-windows-2022-x86-64-avx2.exe")
lc0_Path = Path(rf"C:\Users\cianw\Chess Engines\lc0-v0.29.0-windows-gpu-nvidia-cuda\lc0.exe")
 
 
csvFolder = r"E:\ChessData"
pgnName = "lichess_db_standard_rated_2023-06_2000_5m"
pgnIn = Path(rf"{csvFolder}\{pgnName}.csv")
pgnOut = Path(rf"{csvFolder}\{pgnName}_details.tsv")

lichessData = pd.read_csv(pgnIn, nrows=100000)
lichessData['UTC_dateTime'] = pd.to_datetime(lichessData['UTCDate'] + ' ' + lichessData['UTCTime'])
lichessData.describe()

"""
Functions for Feature Extraction
"""

#Function to Extract Every Nth Word starting at position M, up to N*O words.
def extract_nth_words(text, M, N, O=None):
    words = text.split()
    if O is None:
        endIndex = len(words)
    else: 
        endIndex = min(M-1+N*O, len(words))
    result = [words[i] for i in range(M - 1, endIndex, N)]
    return ' '.join(result)


lichessData['moveNumbers'] = lichessData['Moves'].apply(lambda x: extract_nth_words(x, 1, 3))
lichessData['moveCount'] = lichessData['moveNumbers'].str.split().str.len()
lichessData['whiteMoves_5'] = lichessData['Moves'].apply(lambda x: extract_nth_words(x, 2, 3, 5))
lichessData['blackMoves_5'] = lichessData['Moves'].apply(lambda x: extract_nth_words(x, 3, 3, 5))
lichessData['whiteMoves_10'] = lichessData['Moves'].apply(lambda x: extract_nth_words(x, 2, 3, 10))
lichessData['blackMoves_10'] = lichessData['Moves'].apply(lambda x: extract_nth_words(x, 3, 3, 10))
lichessData['whiteMoves_20'] = lichessData['Moves'].apply(lambda x: extract_nth_words(x, 2, 3, 20))
lichessData['blackMoves_20'] = lichessData['Moves'].apply(lambda x: extract_nth_words(x, 3, 3, 20))


moveTest = "1. e4 e5 2. Nf3 Nc6 3. Bc4 d6 4. h3 Nf6 5. Nc3 Be6 6. Be2 Qd7 7. O-O O-O-O 8. Re1 h6 9. d3 Bxh3 10. gxh3 Qxh3 11. Bf1 Qg4+ 12. Bg2 h5 13. Nh2 Qe6 14. Be3 g5 15. d4 exd4 16. Bxd4 h4 17. Nd5 Bg7 18. Nxf6 Bxf6 19. Bxf6 Qxf6 20. Bh3+ Kb8 21. c3 Rdg8 22. Ng4 Qe7 23. b4 Ne5 24. Re3 a6 25. Kg2 Rg6 26. Qd4 Rf8 27. Nxe5 dxe5 28. Qc4 g4 29. Rd1 gxh3+ 30. Kxh3 Qg5 31. Qf1 f5 32. exf5 Qxf5+ 33. Kh2 Qf4+ 34. Kh1 Qxe3 35. fxe3 Rxf1+ 36. Rxf1 Rg3 37. Re1 e4 38. Kh2 Kc8 39. Re2 Kd7 40. c4 Ke6 41. a4 Kf5 42. Kh1 Kg4 43. Kh2 Kf3 44. Re1 Kf2 45. Rd1 Kxe3 46. Rd7 Kf2 47. Rxc7 e3 48. Rf7+ Ke1 49. Rxb7 e2 50. Re7 Kf2 51. b5 axb5 52. cxb5 Re3 53. Rf7+ Ke1 54. b6 Kd2 55. Rd7+ Kc3 56. Rc7+ Kb4 57. Rc1 e1=Q 58. Rxe1 Rxe1"

engineStockfish = chess.engine.SimpleEngine.popen_uci(stockfish_Path)
engineLc0 = chess.engine.SimpleEngine.popen_uci(lc0_Path)



def evaluateGame(games, engine1=None, engine2 = None):
    if engine1==None:
        print("Please input at least 1 engine for analysis")
        return
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
        info1 = engine1.analyse(board, limit=chess.engine.Limit(time=0.01, depth=20))
        score1 = info1['score'].white()
        #check for mate and assign high value either way, maths says infinity but might not be suitable for analysis later
        if score1.is_mate():
            numericScore1 = float(10000) if score1.mate() > 0 else float(-10000)
        else:
            numericScore1 = score1.cp / 100.0
        if engine2!=None:
            info2 = engine2.analyse(board, limit=chess.engine.Limit(time=0.01, depth=20))
            score2 = info2['score'].white()
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
        #print(f"Score: {info['score']},\tScore (white): " + f"{info['score'].white()},\tDepth: {info['depth']}")
        board.push(move)
    return evalList1, depthList1, evalList2, depthList2, evalDiff, depthDiff
start_time = time.time()
lichessData[['stockfish_eval','stockfish_depth', 
             'Lc0_eval','Lc0_depth', 
             'evalDiff','depthDiff']] = lichessData.apply(evaluateGame,
                                                          engine1=engineStockfish,
                                                          engine2=engineLc0, 
                                                          axis=1, 
                                                          result_type='expand')
print("--- %s seconds ---" % (time.time() - start_time))                                                         




lichessData.to_csv(pgnOut, sep="\t")

lichessDate_read = pd.read_csv(pgnOut, sep="\t")



















engine.close()

def evaluateGame(games):
    #Create a new chess game from the PGN moves
    game = chess.pgn.read_game(io.StringIO(games['Moves']))
    
    # Create a list to store the evaluations for each move
    move_evaluations = []
    
    node = game
    while node.variations:
       next_node = node.variations[0]
       eval_score = next_node.eval()
       if eval_score is not None:
           move_evaluations.append(eval_score.score())
       else:
           move_evaluations.append(0)  # Default value when evaluation is not available
       node = next_node
       
 
    return move_evaluations   
    
    for move in game.mainline_moves():
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(time=0.1))
        evaluation.append(info)

    # Loop through each move in the game and get its evaluation
    node = game
    while node.variations:
        next_node = node.variations[0]
        eval_score = next_node.eval().score()
        move_evaluations.append(eval_score)
        node = next_node



# Assume your DataFrame is named df
lichessData['move_evaluations'] = lichessData.apply(evaluateGame, axis=1, result_type='expand')
    






















# Create a Stockfish engine instance with a specific path to the executable
engine = stockfish.Stockfish(stockfishEngine_Path)

# Set the Stockfish parameters
engine.set_depth(20) # search depth
engine.set_skill_level(20) # skill level

game = io.StringIO("1. e4 { [%clk 0:10:00] } 1... e6 { [%clk 0:10:00] } 2. Nf3 { [%clk 0:09:58] } 2... Nc6 { [%clk 0:09:58] } 3. d4 { [%clk 0:09:55] } 3... d5 { [%clk 0:09:55] } 4. Nc3 { [%clk 0:09:46] } 4... Bb4 { [%clk 0:09:51] } 5. e5 { [%clk 0:09:39] } 5... Bxc3+ { [%clk 0:09:50] } 6. bxc3 { [%clk 0:09:36] } 6... Nge7 { [%clk 0:09:48] } 7. c4 { [%clk 0:09:14] } 7... O-O { [%clk 0:09:42] } 8. cxd5 { [%clk 0:09:04] } 8... Qxd5 { [%clk 0:09:37] } 9. Bd3 { [%clk 0:08:58] } 9... Qd8 { [%clk 0:09:34] } 10. O-O { [%clk 0:08:55] } 10... Nb4 { [%clk 0:09:29] } 11. Bg5 { [%clk 0:08:50] } 11... Qe8 { [%clk 0:09:22] } 12. Be4 { [%clk 0:08:38] } 12... c6 { [%clk 0:09:20] } 13. c3 { [%clk 0:08:33] } 13... Nbd5 { [%clk 0:09:17] } 14. Qc2 { [%clk 0:08:30] } 14... Ng6 { [%clk 0:09:15] } 15. h4 { [%clk 0:08:13] } 15... f6 { [%clk 0:09:13] } 16. exf6 { [%clk 0:08:09] } 16... Nxf6 { [%clk 0:09:10] } 17. Ne5 { [%clk 0:07:56] } 17... Nxe4 { [%clk 0:09:08] } 18. Qxe4 { [%clk 0:07:50] } 18... Nxe5 { [%clk 0:09:05] } 19. Qxe5 { [%clk 0:07:47] } 19... Rf5 { [%clk 0:09:02] } 20. Qg3 { [%clk 0:07:36] } 20... Qg6 { [%clk 0:08:51] } 21. Qc7 { [%clk 0:07:12] } 21... Rxg5 { [%clk 0:08:45] } 22. Qd8+ { [%clk 0:07:05] } 22... Kf7 { [%clk 0:08:39] } 23. hxg5 { [%clk 0:07:01] } 23... a5 { [%clk 0:08:26] } 24. Rae1 { [%clk 0:06:37] } 24... b5 { [%clk 0:08:22] } 25. Qc7+ { [%clk 0:06:30] } 25... Kf8 { [%clk 0:08:18] } 26. Qxc6 { [%clk 0:06:25] } 26... Rb8 { [%clk 0:08:08] } 27. Qd6+ { [%clk 0:06:13] } 27... Kf7 { [%clk 0:08:02] } 28. Qxb8 { [%clk 0:06:11] } 28... Qxg5 { [%clk 0:07:59] } 29. Qxc8 { [%clk 0:06:06] } 29... h6 { [%clk 0:07:56] } 30. Qc7+ { [%clk 0:06:03] } 30... Kg6 { [%clk 0:07:55] } 31. Rxe6+ { [%clk 0:05:58] } 31... Kh7 { [%clk 0:07:54] } 32. Qxa5 { [%clk 0:05:54] } 32... b4 { [%clk 0:07:51] } 33. Qxb4 { [%clk 0:05:52] } 33... Qd5 { [%clk 0:07:47] } 34. Qb1+ { [%clk 0:05:46] } 34... Kg8 { [%clk 0:07:45] } 35. Re8+ { [%clk 0:05:39] } 35... Kf7 { [%clk 0:07:43] } 36. Rfe1 { [%clk 0:05:35] } 36... Qd7 { [%clk 0:07:34] } 37. R8e7+ { [%clk 0:05:25] } 37... Qxe7 { [%clk 0:07:32] } 38. Rxe7+ { [%clk 0:05:24] } 38... Kxe7 { [%clk 0:07:31] } 39. a4 { [%clk 0:05:22] } 1-0")
gameAnalysis = chess.pgn.read_game(game)


evaluation = []
for move in game.mainline_moves():
    board.push(move)
    info = engine.analyse(board, chess.engine.Limit(time=0.1))
    evaluation.append(info)

white_cpl = 0
black_cpl = 0
white_blunders = 0
black_blunders = 0
white_inaccuracies = 0
black_inaccuracies = 0
white_accuracy = 0
black_accuracy = 0
for i, node in enumerate(gameAnalysis.mainline()):
    if i == 0:
        continue
    parent_eval = evaluation[i-1]['score'].relative.score(mate_score=100000)
    child_eval = evaluation[i]['score'].relative.score(mate_score=100000)
    if node.turn:
        # White move
        white_cpl += abs(child_eval - parent_eval)
        if abs(child_eval) > 500:
            white_blunders += 1
        elif abs(child_eval) > 100:
            white_inaccuracies += 1
        else:
            white_accuracy += 1
    else:
        # Black move
        black_cpl += abs(child_eval - parent_eval)
        if abs(child_eval) > 500:
            black_blunders += 1
        elif abs(child_eval) > 100:
            black_inaccuracies += 1
        else:
            black_accuracy += 1






















"""
#Remove Bot Games
lichessData = lichessData[(lichessData['WhiteTitle']!='BOT') & (lichessData['BlackTitle']!='BOT')]

lichessData = lichessData[~lichessData['Termination'].isin(['Rules infraction', 'Abandoned'])]
"""

numericColumns = ["WhiteElo", "BlackElo", "WhiteRatingDiff", "BlackRatingDiff"]
lichessData[numericColumns] = lichessData[numericColumns].astype(float)

#lichessData['whiteOpening'] = lichessData['Opening'].str.split(':').str[0].str.strip()
#lichessData['blackOpening'] = lichessData['Opening'].str.split(':').str[-1].str.strip()


lichessData['whiteWin'] = lichessData['Result'].str.split('-').str[0]
lichessData['blackWin'] = lichessData['Result'].str.split('-').str[-1]

lichessData['whiteMoves'] = lichessData['Moves'].apply(lambda x: re.split(' ', x)[1::3])
lichessData['blackMoves'] = lichessData['Moves'].apply(lambda x: re.split(' ', x)[2::3])
lichessData['moveNumbers'] = lichessData['Moves'].apply(lambda x: re.split(' ', x)[::3])

lichessData['ratingDiff'] = lichessData['WhiteElo'] - lichessData['BlackElo']

#https://herculeschess.com/aggressive-chess-openings-for-white/
aggroWhiteOpenings = ["Vienna Game", 
                     "Smith Morra Gambit",
                     "Scotch Game",
                     "Calabrese Counter Gambit",
                     "Bird’s Opening",
                     "Lay Down Sacrifice",
                     "King’s Gambit",
                     "Italian Game",
                     "Giuoco Piano",
                     "Ruy Lopez"
                     ]
                 


"""
IDEA?
import pandas as pd
import chess.pgn

# Create a sample DataFrame with a single column of PGN notation
lichessData = pd.DataFrame({"pgn_notation": ["1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. d3 d6 5. c3 Bg4 6. h3 Bh5 7. g4 Bg6 8. Nc3 Nd4 9. e5 dxe5 10. Nxe5 c6",
                                   "1. d4 d5 2. c4 c6 3. Nc3 Nf6 4. e3 g6 5. Nf3 Bg7 6. Be2 O-O 7. O-O Nbd7 8. b3 e6 9. Bb2 dxc4 10. bxc4 c5"]})

# Create new columns for white and black moves
lichessData["white_moves"] = ""
lichessData["black_moves"] = ""

# Iterate through the rows of the DataFrame
for i, row in lichessData.iterrows():
    pgn_notation = row["pgn_notation"]
    # Create a chess.pgn object from the PGN notation
    game = chess.pgn.read_game(pgn_notation)
    # Initialize the white_moves and black_moves strings
    white_moves = ""
    black_moves = ""
    for move in game.main_line():
        if move.parent.turn:
            white_moves += move.uci() + " "
        else:
            black_moves += move.uci() + " "
    # Add the extracted moves to the DataFrame
    lichessData.at[i, "white_moves"] =
"""



















