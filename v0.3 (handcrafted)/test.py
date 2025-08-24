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
## AUTHOR: David Carteau, France, August 2025                               ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## PURPOSE:                                                                 ##
## Test the Nostradamus UCI chess engine                                    ##
##############################################################################

import os
import torch

from train import Dataset, Network, TransformerBlock, Attention, MultiHeadAttention


##############################################################################
## CLASSES SECTION                                                          ##
##############################################################################

class Engine():
    def __init__(self):
        self.path = get_last_model_path()
        self.model = Network().load(self.path)
        self.dataset = Dataset(load=False)
    #end def
    
    def predict(self, positions):
        X, y = self.dataset.get_samples(positions)
        
        self.model.eval()
        
        with torch.no_grad():
            fr, to = self.model(X)
        #end with
            
        fr = torch.softmax(fr, dim=1).argmax(dim=1).numpy()
        to = torch.softmax(to, dim=1).argmax(dim=1).numpy()
        
        moves = []
        
        for i in range(len(positions)):
            if "w" in positions[i]:
                move = self.uci(fr[i], to[i])
            else:
                move = self.uci(fr[i] ^ 56, to[i] ^ 56)
            #end if
            
            moves.append(move)
        #end for
        
        return moves
    #end def
    
    def uci(self, fr, to):
        return "abcdefgh"[fr % 8] + str(1 + fr // 8) + "abcdefgh"[to % 8] + str(1 + to // 8)
    #end def
#end class


##############################################################################
## FUNCTIONS SECTION (utils & main)                                         ##
##############################################################################

def get_last_model_path():
    models = {}
    
    for entry in os.scandir("./models"):
        if entry.is_file() and entry.name.startswith("epoch-") and entry.name.endswith(".pt"):
            epoch = int(entry.name.split("-")[1])
            models[epoch] = entry.name
        #end if
    #end for
    
    last_epoch = max(models.keys())
    last_model = models[last_epoch]
    last_model_path = f'./models/{last_model}'
    
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
    print(f'Nostradamus 0.3 (2025)')
    print("https://github.com/david-carteau")
    print()
    
    engine = Engine()
    
    print(f'Using model: {engine.path}')
    print()
    
    positions = load_data()
    
    moves = engine.predict(positions)
    
    fr_acc, to_acc, mv_acc = 0, 0, 0
    
    for position, move in zip(positions, moves):
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
