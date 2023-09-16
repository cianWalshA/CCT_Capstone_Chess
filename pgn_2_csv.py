# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 21:18:59 2023

@author: cianw


This program will convert chess PGN databases into CSV files.

This particular program has controls for rating and ELO. It will also remove unrated players and games not ending in checkmate, forfeit or time forfeit.

The process here will require chunking due to the size of lichess databases. (200 Gb+ for 1 month)
"""
import sys
import os
import csv
import chess.pgn
import re
import pandas as pd
import seaborn as sns
import datetime
from pathlib import Path

pgnFolder = r"E:\ChessData"
csvFolder = r"E:\ChessData\explorationOutputs"
pgnName = "lichess_db_standard_rated_2023-06"
outputName = "_10MinGames_15Jun2023_limit_10GB"
#FIX MOVE LINES CODE

def csvCommit(outFile, dictToWrite, csvHeaders):
    file_exists = os.path.isfile(outFile)
    with open(outFile, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=csvHeaders)
        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
        for row in dictToWrite:
            writer.writerow(row)
           
def pgn_2_csv_fix_lines(pgnName, pgnFolder, csvFolder, whiteELO=0, blackELO=0 , timeControl=0, save=0, overwrite = 1, dateReturn = None, memoryLimitGB=1, gameDepositLength=10000):

    pgnIn = Path(rf"{pgnFolder}\{pgnName}.pgn")
    pgnOut = Path(rf"{csvFolder}\{pgnName}_{outputName}.csv")
    pgnLabels = ['Event', 'Site', 'Date', 'Round', 'White', 'Black', 'Result', 'UTCDate', 'UTCTime', 'WhiteElo', 'BlackElo', 'WhiteRatingDiff','BlackRatingDiff','WhiteTitle','BlackTitle', 'ECO', 'Opening', 'TimeControl', 'Termination', 'Moves']
    chunkWriter=1
    gameProcessed=1
    gameCounter=1
    memoryLimit = memoryLimitGB*1024*1024*1024
    games = []
    dic = dict.fromkeys(pgnLabels)
    
    if os.path.exists(pgnOut) == True and overwrite == 1:
        os.remove(pgnOut)
        print(f"File {pgnOut} deleted successfully.")
        
    
    with open(pgnIn, "r") as stream:
        for l in stream:
            #This section loops through each line of the PGN and checks what to do with it based on the first character
            if l[0] == "[":
                string = l[1:-1]
                header = string.split()[0]
                dic[header] = string[len(header):-1].strip().strip('"')

            elif l[0] == "1":       
                string = l
                header = 'Moves'
                if string.find("...") or string.find("{"):
                   #Remove Lichess Captions in Moves
                   string = re.sub("\{(.*?)\}", "", string)
                   string = re.sub("(\d+\.{3})", "", string)
                   string = re.sub("\?!|!\?|\?*|!*", "", string)
                   string = re.sub("\s+", " ", string)
                   string = re.sub(" (1\/2-1\/2|\d-\d)", "", string)
                #This section filters out games based on the requirements of the function above.
                if (dic['WhiteElo'] in ['?','-'] 
                    or dic['BlackElo'] in ['?','-'] 
                    or dic['TimeControl'] in ['?','-']
                    or dic['WhiteTitle'] in ['BOT']
                    or dic['BlackTitle'] in ['BOT']
                    or dic['Termination'] in ['Abandoned', 'Rules infraction']
                    ):
                    pass

                elif int(dic['WhiteElo']) >= whiteELO and int(dic['BlackElo']) >= blackELO and int(dic['TimeControl'].split('+')[0]) >= timeControl:
                    dic[header] = string
                    games.append(dic)
                    gameCounter+=1
                    if gameCounter%1000 == 0:
                        print(rf"Games Saved: {gameCounter}")
                else:
                    pass
                
                gameProcessed+=1
                if dateReturn != None:
                    if (datetime.datetime.strptime(dic["UTCDate"], "%Y.%m.%d")) == dateReturn :
                        return print(rf"Date limit of {dateReturn} reached")
                if gameProcessed%gameDepositLength == 0:
                    print(rf"Games Processed: {gameProcessed}")
                dic = dict.fromkeys(pgnLabels)
                
                
            else:
                pass
            
            #Commits the "games" Dictionary to the CSV 
            if (len(games) >= gameDepositLength):
                csvCommit(pgnOut, games, pgnLabels)
                chunkWriter +=1
                print(chunkWriter)
                games = []
            if os.path.exists(pgnOut) == True:
                if pgnOut.stat().st_size >= memoryLimit:
                    return print(rf"Memory limit of {memoryLimit} reached")
        
        csvCommit(pgnOut, games, pgnLabels)
            
    #    del pgn, games, dic, pgnIn, pgnOut, contents, string, header, l
    

#Create DF from PGN (PERSONAL GAME NOTATION)
pgn_2_csv_fix_lines(pgnName, pgnFolder, csvFolder, whiteELO=0, blackELO=0, timeControl=600, save = 1, overwrite=1, dateReturn=(datetime.date(2023, 6,15)),  memoryLimitGB = 10, gameDepositLength= 100000)
