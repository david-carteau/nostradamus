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
## NAME: train_tokenizer.py                                                 ##
## AUTHOR: David Carteau, France, October 2024                              ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## IMPORTANT !                                                              ##
## This chess engine is very weak: its only purpose is to see how language  ##
## models can be trained to play chess ;-)                                  ##
##############################################################################


from utils import transform

from tokenizers import (
    models,
    pre_tokenizers,
    decoders,
    Tokenizer
)

from transformers import PreTrainedTokenizerFast

MAX_LENGTH = 82


def train_tokenizer(save_path="./model"):
    chars = [c for c in ".PNBRQKpnbrqkacdefgh/-sS"]
    
    squares = []
    
    for row in range(8):
        for col in range(8):
            square = "abcdefgh"[col] + "12345678"[row]
            squares.append(square)
        #end for
    #end for
    
    specials = ["<s>", "</s>", "<pad>", "<unk>"]
    
    tokens = chars + squares + specials
    
    vocab = {token: i for i, token in enumerate(tokens)}
    
    tokenizer = Tokenizer(models.WordPiece(vocab=vocab, unk_token="<unk>"))
    
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    tokenizer.decoder = decoders.WordPiece()
    
    wrapped_tokenizer = PreTrainedTokenizerFast(
        model_max_length=MAX_LENGTH,
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        clean_up_tokenization_spaces=True
    )
    
    wrapped_tokenizer.save_pretrained(save_path)
    
    # test with startpos
    
    line = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - d2d4\n"
    
    position = transform(line)
    
    print(position)
    
    ids = wrapped_tokenizer.encode(position)
    
    print(ids)
    print("length =", len(ids))
    
    assert len(ids) == MAX_LENGTH
#end def

if __name__ == "__main__":
    train_tokenizer()
#end if
