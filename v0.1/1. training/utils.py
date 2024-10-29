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
## NAME: utils.py                                                           ##
## AUTHOR: David Carteau, France, October 2024                              ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## IMPORTANT !                                                              ##
## This chess engine is very weak: its only purpose is to see how language  ##
## models can be trained to play chess ;-)                                  ##
##############################################################################


def transform(line):
    fen, stm, cas, enp, mov = line.strip().split()
    
    for i in range(1, 9):
        fen = fen.replace(str(i), "." * i)
    #end for
    
    stm = {"w": "S", "b": "s"}[stm]
    
    cas = ("K" if "K" in cas else "-") + ("Q" if "Q" in cas else "-") + ("k" if "k" in cas else "-") + ("q" if "q" in cas else "-")
    
    fr = mov[0:2]
    to = mov[2:4]
    
    if len(mov) == 5:
        pr = mov[4]
    else:
        pr = "-"
    #end if
    
    line = f'{fen}{stm}{cas}'
    line = " ".join([c for c in line])
    line = f'<s> {line} {enp} {fr} {to} {pr} </s>'
    
    return line
#end def
