"""
The Nostradamus UCI chess engine
Copyright (c) 2024-2025, by David Carteau. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

##############################################################################
## NAME: nostradamus.py                                                     ##
## AUTHOR: David Carteau, France, September 2025                            ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## PURPOSE:                                                                 ##
## The Nostradamus UCI chess engine !!!                                     ##
##############################################################################

import chess
import random

from test import Engine


##############################################################################
## FUNCTIONS SECTION (utils & main)                                         ##
##############################################################################

def seed():
    random.seed(0)
#end def


def send(message):
    print(message, flush=True)
#end def


def main():
    engine = Engine()
    name, author = engine.infos()
    
    seed()
    stats = {}
    
    while True:
        command = input().strip()
        
        if command.startswith("quit"):
            break
        #end if
        
        if command.startswith("ucinewgame"):
            seed()
            stats = {}
            continue
        #end if
        
        if command.startswith("uci"):
            send("id name " + name)
            send("id author " + author)
            send("option name Hash type spin default 0 min 0 max 0")
            send("uciok")
            continue
        #end if
        
        if command.startswith("isready"):
            send("readyok")
            continue
        #end if
        
        if command.startswith("position fen "):
            idx = command.find("moves")
            fen = command[13:idx]
            board = chess.Board(fen)
        #end if
        
        if command.startswith("position startpos"):
            board = chess.Board()
            moves = []
        #end if
        
        if command.startswith("position ") and " moves " in command:
            command = command.split()
            idx = command.index("moves")
            moves = command[idx+1:]
            
            for move in moves:
                move = chess.Move.from_uci(move)
                board.push(move)
            #end for
            
            continue
        #end if
        
        if command.startswith("go"):
            legal_moves = [m.uci() for m in board.legal_moves]
            
            position = board.fen()
            position = position.split()[:4]
            position = " ".join(position) + " a1a1"
            
            [move] = engine.predict([position])
            
            if move not in legal_moves:
                send(f'info string Are you kidding, Nostradamus?! "{move}" is not a legal move!')
                move = random.choice(legal_moves)
                stats["random"] = stats.get("random", 0) + 1
            #end if
            
            stats["total"] = stats.get("total", 0) + 1
            
            rnd = stats.get("random", 0)
            tot = stats["total"]
            pct = 100.0 * (rnd / tot)
            
            send(f'bestmove {move}')
            send(f'info string {rnd}/{tot} random moves ({pct:0.2f}%)')
            
            continue
        #end if
    #end while
#end if


if __name__ == "__main__":
    main()
#end if
