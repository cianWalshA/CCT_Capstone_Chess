

import sys
import io
import os
import csv
import chess
import chess.pgn
import stockfish
import re
import pandas as pd
import seaborn as sns
from pathlib import Path
from datetime import datetime 

stockfishEngine_Path = Path(rf"C:\Users\cianw\stockfish_15.1_win_x64_avx2\stockfish-windows-2022-x86-64-avx2.exe")
 
csvFolder = r"C:\Users\cianw\Documents\dataAnalytics\projectFinal\Data\Chess\Lichess_CSV"
pgnName = "lichess_db_standard_rated_2022-11"
pgnOut = Path(rf"{csvFolder}\{pgnName}.csv")

lichessData = pd.read_csv(pgnOut, nrows=10000)
lichessData['UTC_dateTime'] = pd.to_datetime(lichessData['UTCDate'] + ' ' + lichessData['UTCTime'])
lichessData.describe()


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



















