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
## NAME: test.py                                                            ##
## AUTHOR: David Carteau, France, September 2025                            ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## PURPOSE:                                                                 ##
## Test the Nostradamus UCI chess engine                                    ##
##############################################################################

import os
import sys
import chess
import torch
import train


##############################################################################
## CLASSES SECTION                                                          ##
##############################################################################

class Engine():
    def __init__(self):
        self.path = get_last_model_path()
        self.model = train.Network().load(self.path)
        self.dataset = train.Dataset(load=False)
    #end def
    
    def infos(self):
        return train.infos()
    #end def
    
    def predict(self, positions):
        boards, moves = self.dataset.get_samples(positions)
        
        self.model.eval()
        
        with torch.no_grad():
            sequences = boards
            
            fr = self.model(sequences)
            fr = torch.softmax(fr, dim=1).argmax(dim=1).view(-1,1)
            
            sequences = torch.cat((boards, fr), dim=1)
            
            to = self.model(sequences)
            to = torch.softmax(to, dim=1).argmax(dim=1).view(-1,1)
            
            sequences = torch.cat((boards, fr, to), dim=1)
            
            up = self.model(sequences)
            up = torch.softmax(up, dim=1).argmax(dim=1).view(-1,1)
        #end with
        
        fr = fr.flatten().numpy()
        to = to.flatten().numpy()
        up = up.flatten().numpy()
        
        moves = []
        
        for i in range(len(positions)):
            if "w" in positions[i]:
                move = self.uci(fr[i], to[i], up[i])
            else:
                move = self.uci(fr[i] ^ 56, to[i] ^ 56, up[i])
            #end if
            
            moves.append(move)
        #end for
        
        return moves
    #end def
    
    def uci(self, fr, to, up):
        move  = "abcdefgh"[fr % 8] + str(1 + fr // 8)
        move += "abcdefgh"[to % 8] + str(1 + to // 8)
        move += " nbrq"[up]
        
        return move.strip()
    #end def
#end class


##############################################################################
## FUNCTIONS SECTION (utils & main)                                         ##
##############################################################################

def get_last_model_path():
    # if pyinstaller is used, base_path = sys._MEIPASS
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = '.'
    #end if
    
    models = []
    
    for entry in os.scandir(f'{base_path}/models'):
        if entry.is_file() and entry.name.startswith("epoch-") and entry.name.endswith(".pt"):
            models.append(entry.name)
        #end if
    #end for
    
    last_model = sorted(models)[-1]
    last_model_path = f'{base_path}/models/{last_model}'
    
    return last_model_path
#end def


def load_data():
    positions = []
    
    with open('./positions/positions-shuffled.txt', 'rt') as lines:
        for line in lines:
            positions.append(line.strip())
            
            if len(positions) >= (16 * 1024):
                break
            #end if
        #end for
    #end with
    
    return positions
#end def


def main():
    engine = Engine()
    engine.infos()
    
    print(f'Using model: {engine.path}')
    print()
    
    positions = load_data()
    
    moves = engine.predict(positions)
    
    legal, fr_acc, to_acc, mv_acc = 0, 0, 0, 0
    
    with open('illegal_moves.txt', 'wt') as file:
        for position, move in zip(positions, moves):
            fen = ' '.join(position.split()[:-1]) + ' 0 1'
            board = chess.Board(fen)
            legal_moves = [m.uci() for m in board.legal_moves]
            
            if move in legal_moves:
                legal += 1
            else:
                file.write(f'{fen[:-4]} {move}\n')
            #end if
            
            fr_true = position.split()[-1][:2]
            to_true = position.split()[-1][2:]
            fr_pred = move[:2]
            to_pred = move[2:]
            
            if fr_pred == fr_true:
                fr_acc += 1
            #end if
            
            if to_pred == to_true:
                to_acc += 1
            #end if
            
            if fr_pred == fr_true and to_pred == to_true:
                mv_acc += 1
            #end if
        #end for
    #end with
    
    print('legal         :', 100.0 * legal  / len(positions))
    print('accuracy fr   :', 100.0 * fr_acc / len(positions))
    print('accuracy to   :', 100.0 * to_acc / len(positions))
    print('accuracy fr+to:', 100.0 * mv_acc / len(positions))
    print()
    
    print("Testing complete!")
    print()
#end def


if __name__ == "__main__":
    main()
#end if
