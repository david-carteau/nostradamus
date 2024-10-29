"""
The Nostradamus UCI chess engine
Copyright (c) 2024, by David Carteau. All rights reserved.

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
## NAME: nostradamus.py (UCI chess engine)                                  ##
## AUTHOR: David Carteau, France, October 2024                              ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## IMPORTANT !                                                              ##
## This chess engine is very weak: its only purpose is to see how language  ##
## models can be trained to play chess ;-)                                  ##
##############################################################################


import chess
import random

from utils import transform
from transformers import pipeline, set_seed


MAX_LENGTH = 82

ENGINE_NAME   = "Nostradamus 0.1"
ENGINE_AUTHOR = "David Carteau"
ENGINE_URL    = "https://github.com/david-carteau"

def send(message):
    print(message, flush=True)
#end def


def think(line,generator):
    print(line)
    position = transform(line)
    
    position = position.split()[:-4]
    
    position = " ".join(position)
    
    generation = generator(position, max_new_tokens=4, truncation=True, num_beams=5)
    generation = generation[0]["generated_text"].strip()
    
    move = generation[len(position):].split()
    
    return "".join(move).rstrip("-").lower()
#end def


def main():
    print(ENGINE_NAME)
    print("UCI chess engine by", ENGINE_AUTHOR, "(2024)")
    print("https://github.com/david-carteau/nostradamus")
    print()
    
    stats = {}
    set_seed(0)
    random.seed(0)
    
    generator = pipeline("text-generation", tokenizer="./model", model="./model-0", device="cpu")
    
    while True:
        command = input().strip()
        
        if command.startswith("quit"):
            break
        #end if
        
        if command.startswith("ucinewgame"):
            stats = {}
            set_seed(0)
            random.seed(0)
            continue
        #end if
        
        if command.startswith("uci"):
            send("id name " + ENGINE_NAME)
            send("id author " + ENGINE_AUTHOR)
            send("option name Hash type spin default 0 min 0 max 0")
            send("uciok")
            continue
        #end if
        
        if command.startswith("isready"):
            send("readyok")
            continue
        #end if
        
        if command.startswith("position fen "):
            fen = command[13:]
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
            
            line = board.fen()
            line = line.split()[:4]
            line = " ".join(line) + " ----"
            
            move = think(line, generator)
            
            if move not in legal_moves:
                send(f'info string Nostradamus, are you kidding ?! {move} is not a legal move !')
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
