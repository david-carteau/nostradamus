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
## NAME: prepare_data.py                                                    ##
## AUTHOR: David Carteau, France, October 2024                              ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## IMPORTANT !                                                              ##
## This chess engine is very weak: its only purpose is to see how language  ##
## models can be trained to play chess ;-)                                  ##
##############################################################################


def main():
    positions = {}
    
    while True:
        try:
            line = input().strip()
            
            if line.startswith("["):
                continue
            #end if
            
            if len(line) == 0:
                position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            #end if
            
            if line.startswith("{"):
                position = line[2:-2]
            #end if
            
            if 0 < len(line) <= 5:
                position = position.split()[:-2]
                position = " ".join(position)
                
                move = line
                
                if "-" in move:
                    continue
                #end if
                
                if position not in positions:
                    positions[position] = {}
                #end if
                
                positions[position][move] = positions[position].get(move, 0) + 1
            #end if
        except:
            break
        #end try
    #end while
    
    with open("samples.fen", "wt") as o_file:
        for position in positions:
            moves = positions[position]
            moves = {move: count for move, count in sorted(moves.items(), key=lambda item: item[1])}
            
            move = list(moves.keys())[-1]
            o_file.write(f'{position} {move}\n')
        #end for
    #end with
#end if


if __name__ == "__main__":
    main()
#end if
