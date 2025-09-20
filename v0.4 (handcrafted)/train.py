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
## NAME: train.py                                                           ##
## AUTHOR: David Carteau, France, September 2025                            ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## PURPOSE:                                                                 ##
## Train the neural network (input: fenstring, output: best move to play)   ##
##############################################################################

##############################################################################
## This work was largely inspired by Andrej Karpathy's excellent tutorial:  ##
## - https://www.youtube.com/watch?v=kCc8FmEb1nY                            ##
## - https://github.com/karpathy/nanoGPT                                    ##
##                                                                          ##
## A big thank you to Andrej for sharing this invaluable material! David.   ##
##############################################################################

import os
import gzip
import json
import torch
import pickle
import random

from tqdm import tqdm


##############################################################################
## VARIABLES SECTION (adjust if needed)                                     ##
##############################################################################

# engine identification
ENGINE_NAME   = "Nostradamus 0.4"
ENGINE_AUTHOR = "By David Carteau (2025)"
ENGINE_URL    = "https://github.com/david-carteau"

# network architecture
N_FEATURES = 256
N_HEADS    = 8
N_LAYERS   = 8

# projection factor (feed forward layers)
PROJECTION_FACTOR = 2.0

# learning rate scheduler
LR = [0.0005] * 3

# batch size
# if you change this value and encountered problems,
# remove the ./cache folder and/or delete all of its content
BATCH_SIZE = 2 * 1024

# weight decay, i.e. try to reduce the magnitude of weights and biases
DECAY = 1e-5

# random seed
# 21.05.2014 = date of Orion's first public release :-)
SEED = 21052014

# device used for training
# set to "auto" for automatic selection
# set to "cuda" for Nvidia GPU, "mps" for Apple Silicon, etc.
TRAINING_DEVICE = "auto"

# maximum number of batches by epoch
N_BATCHES_BY_EPOCH = None


##############################################################################
## CONSTANTS SECTION (do not modify)                                        ##
##############################################################################

# number of epochs
N_EPOCHS = len(LR)

# Q value (+/- 1.98)
# Q and -Q are the minimum and maximum values allowed for weights and biases
# this opens the future possibility to quantize the network (post-training)
# (inherited from the Cerebrum library)
Q = 127 / 64

# structure of folders
CACHE_PATH = "./cache"
MODELS_PATH = "./models"
POSITIONS_PATH = "./positions"

# dataset file name
POSITIONS_FILE = "positions-shuffled.txt"

# device used for the training
if TRAINING_DEVICE == "auto":
    if torch.mps.is_available():
        TRAINING_DEVICE = "mps"
    elif torch.cuda.is_available():
        TRAINING_DEVICE = "cuda"
    else:
        TRAINING_DEVICE = "cpu"
    #end if
#end if


##############################################################################
## CLASSES SECTION                                                          ##
##############################################################################

# dataset loader
class Dataset():
    def __init__(self, load=True):
        if not load:
            return
        #end if
        
        dataset = f'{POSITIONS_PATH}/{POSITIONS_FILE}'
        
        lines = []
        batch = 0
        
        print("Reading dataset...")
        
        with open(dataset, 'rt') as file:
            for line in tqdm(file, unit_scale=True):
                lines.append(line)
                
                if len(lines) == BATCH_SIZE:
                    path = f'{CACHE_PATH}/{BATCH_SIZE}/batch.{batch}.pickle'
                    
                    if not os.path.exists(path):
                        self.save_batch(batch, lines)
                    #end if
                    
                    lines = []
                    batch += 1
                #end if
            #end for
        #end with
        
        lines = None
        
        print()
        
        self.batches = [i for i in range(batch)]
    #end def
    
    def __len__(self):
        return len(self.batches)
    #end def
    
    def __iter__(self):
        self.batch = -1
        random.shuffle(self.batches)
        
        return self
    #end def
    
    def __next__(self):
        self.batch += 1
        
        if self.batch == len(self.batches):
            raise StopIteration
        #end if
        
        batch = self.batches[self.batch]
        
        return self.load_batch(batch)
    #end def
    
    # get board and move from each line of POSITIONS_FILE
    def get_sample(self, line):
        fen, stm, cas, enp, move = line.strip().split()
        
        rows = fen.split("/")
        
        assert len(rows) == 8
        
        square = 0
        
        pieces = "pnbrqk"
        digits = "12345678"
        
        if stm == "w":
            stm_pieces = pieces.upper()
        else:
            stm_pieces = pieces
        #end if
        
        board = [0] * 64
        
        for row in rows[::-1]:
            for char in row:
                if char in digits:
                    square += digits.find(char) + 1
                else:
                    if stm == "w":
                        sq = square
                    else:
                        sq = square ^ 56
                    #end if
                    
                    idx = pieces.find(char.lower())
                    
                    if char in stm_pieces:
                        board[sq] = (2 * idx) + 1 # odd number
                    else:
                        board[sq] = (2 * idx) + 2 # even number
                    #end if
                    
                    square += 1
                #end if
            #end for
        #end for
        
        assert square == 64
        
        fr = "abcdefgh".index(move[0]) + 8 * "12345678".index(move[1])
        to = "abcdefgh".index(move[2]) + 8 * "12345678".index(move[3])
        
        if stm == "b":
            fr ^= 56
            to ^= 56
        #end if
        
        # up = pawn's promotion
        if len(move) == 5:
            up = pieces.index(move[4].lower())
        else:
            up = 0
        #end if
        
        move = [fr, to, up]
        
        return board, move
    #end def
    
    def get_samples(self, lines):
        boards = []
        moves = []
        
        for line in lines:
            try:
                board, move = self.get_sample(line)
            except Exception as e:
                print("Error", e, "->", line.strip())
                continue
            #end try
            
            boards.append(board)
            moves.append(move)
        #end for
        
        X = torch.tensor(boards)
        y = torch.tensor(moves)
        
        return X, y
    #end def
    
    # load a batch from disk
    def load_batch(self, batch):
        name = f'{CACHE_PATH}/{BATCH_SIZE}/batch.{batch}.pickle'
        
        with gzip.GzipFile(name, "rb") as file:
            return pickle.load(file)
        #end with
    #end def
    
    # save a batch to disk
    def save_batch(self, batch, lines):
        name = f'{CACHE_PATH}/{BATCH_SIZE}/batch.{batch}.pickle'
        
        X, y = self.get_samples(lines)
        
        with gzip.GzipFile(name, "wb", compresslevel=6) as file:
            pickle.dump((X, y), file)
        #end with
    #end def
#end class


# batched multi-head attention
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_features, n_heads):
        super().__init__()
        
        C = n_features
        self.n_heads = n_heads
        
        self.attn = torch.nn.Linear(C, 3 * C, bias=False)
        #self.proj = torch.nn.Linear(C, C, bias=False)
    #end def
    
    def forward(self, x):
        B, T, C = x.size()
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.attn(x).split(C, dim=2)
        
        # head size
        H = k.size(-1)
        
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, n_heads, T, H)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, n_heads, T, H)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, n_heads, T, H)
        
        if TRAINING_DEVICE == "cuda":
            # dropout is set to 0.0 to match the code used for non-CUDA devices
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        else:
            # (B, T, H) @ (B, H, T) --> (B, T, T)
            w = q @ k.transpose(-2, -1) * H**-0.5
            w = torch.softmax(w, dim=-1)
            
            # (B, T, T) @ (B, T, H) --> (B, T, H)
            y = w @ v
        #end if
        
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # output projection
        #y = self.proj(y)
        
        return y
    #end def
#end class


# transformer block (encoder only)
class TransformerBlock(torch.nn.Module):
    def __init__(self, n_features, n_heads, config=['att', 'att_norm', 'ffd', 'ffd_norm']):
        super().__init__()
        
        self.att = None
        self.att_norm = None
        self.ffd = None
        self.ffd_norm = None
        
        if 'att' in config:
            self.att = torch.nn.Sequential(
                MultiHeadAttention(n_features, n_heads),
                torch.nn.Dropout(0.1)
            )
        #end if
        
        if 'att_norm' in config:
            self.att_norm = torch.nn.LayerNorm(n_features, bias=False)
        #end if
        
        if 'ffd' in config:
            n_projection = int(n_features * PROJECTION_FACTOR)
            
            self.ffd = torch.nn.Sequential(
                torch.nn.Linear(n_features, n_projection),
                torch.nn.ReLU(),
                torch.nn.Linear(n_projection, n_features),
                torch.nn.Dropout(0.1)
            )
        #end if
        
        if 'ffd_norm' in config:
            self.ffd_norm = torch.nn.LayerNorm(n_features, bias=False)
        #end if
    #end def
    
    def forward(self, x):
        if self.att is not None:
            x = x + self.att(x)
        #end if
        
        if self.att_norm is not None:
            x = self.att_norm(x)
        #end if
        
        if self.ffd is not None:
            x = x + self.ffd(x)
        #end if
        
        if self.ffd_norm is not None:
            x = self.ffd_norm(x)
        #end if
        
        return x
    #end def
#end class


# engine model (input: board, output: move)
class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # there can be 77 different token_ids in a sequence
        # 0: empty square, 1-12: types of piece (for 'board')
        # 13-76: 64 squares (for 'fr' and 'to')
        n_token_ids = 77
        
        # a sequence can be composed up to 66 different tokens
        # 0-63: 64 squares, 64: 'fr', 65: 'to'
        max_seq_len = 66
        
        self.tok_embeddings = torch.nn.Embedding(n_token_ids, N_FEATURES)
        self.pos_embeddings = torch.nn.Embedding(max_seq_len, N_FEATURES)
        
        layers = []
        
        for _ in range(N_LAYERS):
            layer = TransformerBlock(n_features=N_FEATURES, n_heads=N_HEADS)
            layers.append(layer)
        #end for
        
        self.layers = torch.nn.ModuleList(layers)
        
        self.head_fr = torch.nn.Linear(N_FEATURES, 64)
        self.head_to = torch.nn.Linear(N_FEATURES, 64)
        self.head_up = torch.nn.Linear(N_FEATURES, 5)
    #end def
    
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    #end def
    
    def forward(self, sequences):
        B, T = sequences.shape
        
        if T >= 65:
            # 'fr' token_ids need to be adjusted to avoid confusion with types of piece token_ids
            sequences[:,64] += 13
        #end if
        
        if T >= 66:
            # 'to' token_ids need to be adjusted to avoid confusion with types of piece token_ids
            sequences[:,65] += 13
        #end if
        
        x = self.tok_embeddings(sequences) + self.pos_embeddings(torch.arange(T, device=sequences.device))
        
        for layer in self.layers:
            x = layer(x)
        #end for
        
        x = x[:,-1,:]
        
        board = sequences[:,:64]
        
        if T == 64:
            # input: 'board' --> predict the 'fr' part of the move
            x = self.head_fr(x)
            
            # 'fr' squares that are empty or with an opponent piece are neutralised
            x = torch.where((board % 2) == 0, float("-inf"), x)
        #end if
        
        if T == 65:
            # input: 'board' + 'fr' --> predict the 'to' part of the move
            x = self.head_to(x)
            
            # 'to' squares with a piece from the side-to-move player are neutralised
            x = torch.where((board % 2) == 1, float("-inf"), x)
        #end if
        
        if T == 66:
            # input: 'board' + 'fr' + 'to' --> predict the 'up' part of the move
            x = self.head_up(x)
        #end if
        
        return x
    #end def
    
    def load(self, model_path):
        state_dict = torch.load(model_path, weights_only=True)
        self.load_state_dict(state_dict)
        
        return self
    #end def
    
    def save(self, epoch, loss):
        model_path = f'{MODELS_PATH}/epoch-{epoch + 1}-{format(loss)}.pt'
        state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
        
        torch.save(state_dict, model_path)
        
        model_path = model_path[:-3] + '.json'
        state_dict = {k: v.tolist() for k, v in state_dict.items()}
        
        with open(model_path, 'wt') as file:
            json.dump(state_dict, file, indent=4)
        #end with
    #end def
#end class


# engine trainer
class Trainer():
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = Network().to(TRAINING_DEVICE)
        
        print("PyTorch", torch.__version__)
        print("Using device:", TRAINING_DEVICE)
        print()
        
        print(f'Model has {self.model.n_params():,} parameters')
        print()
        
        lr_lambda = lambda epoch: LR[epoch] if epoch < len(LR) else 0
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0, weight_decay=DECAY)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
    #end def
    
    def train_one_batch(self, X, y):
        # move data to the training device (i.e. CPU or GPU)
        X_true, y_true = X.to(TRAINING_DEVICE), y.to(TRAINING_DEVICE)
        
        # predict 3 tokens in a row ('fr', 'to', 'up')
        loss_cum = 0
        
        # re-init gradients
        self.optimizer.zero_grad()
        
        # input: 'board' --> output: 'fr'
        sequences = X_true
        
        y_true_fr = y_true[:,0]
        y_pred_fr = self.model(sequences)
        loss = self.criterion(y_pred_fr, y_true_fr)
        loss.backward()
        loss_cum += loss.item()
        
        # input: 'board' + 'fr' --> output: 'to'
        sequences = torch.cat((X_true, y_true[:,:1]), dim=1)
        
        y_true_to = y_true[:,1]
        y_pred_to = self.model(sequences)
        loss = self.criterion(y_pred_to, y_true_to)
        loss.backward()
        loss_cum += loss.item()
        
        # input: 'board' + 'fr' + 'to' --> output: 'up'
        sequences = torch.cat((X_true, y_true[:,:2]), dim=1)
        
        y_true_up = y_true[:,2]
        y_pred_up = self.model(sequences)
        loss = self.criterion(y_pred_up, y_true_up)
        loss.backward()
        loss_cum += loss.item()
        
        # update model parameters
        self.optimizer.step()
        
        # clip model weights (in the perspective of quantization)
        with torch.no_grad():
            for param in self.model.parameters():
                param.clamp_(-Q, Q)
            #end for
        #end with
        
        return loss_cum
    #end def
    
    def train_one_epoch(self, epoch):
        loss = 0
        losses = []
        
        rate = self.scheduler.get_last_lr()[0]
        
        n_batches = 0
        
        # iterate through the dataset
        for X, y in (pbar := tqdm(self.dataset)):
            loss = self.train_one_batch(X, y)
            losses.append(loss)
            
            loss = losses[-32:]
            loss = sum(loss) / len(loss)
            
            pbar.set_postfix({"LR:": format(rate), "Loss:": format(loss)})
            
            n_batches += 1
            
            if N_BATCHES_BY_EPOCH is not None and n_batches >= N_BATCHES_BY_EPOCH:
                break
            #end if
        #end for
        
        # update learning rate
        self.scheduler.step()
        
        # save model
        self.model.save(epoch, loss)
    #end def
#end class


##############################################################################
## FUNCTIONS SECTION (utils & main)                                         ##
##############################################################################

# engine identification
def infos():
    print(ENGINE_NAME)
    print(ENGINE_AUTHOR)
    print(ENGINE_URL)
    print()
    
    return ENGINE_NAME, ENGINE_AUTHOR
#end def


# pretty formatting of float values
def format(n):
    n = f"{n:.05f}"
    
    if n == "-0.00000":
        n = "0.00000"
    #end if
    
    return n
#end def


# main function
def main():
    # welcome message
    infos()
    
    # use 16-bit precision to speed-up training and save memory
    torch.set_default_dtype(torch.bfloat16)
    
    # set seed for reproducibilty
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.mps.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    # clear cache
    if TRAINING_DEVICE == "mps":
        torch.mps.empty_cache()
    elif TRAINING_DEVICE == "cuda":
        torch.cuda.empty_cache()
    #endif
    
    # creation of folders structure
    for path in [CACHE_PATH, f'{CACHE_PATH}/{BATCH_SIZE}', MODELS_PATH]:
        if not os.path.exists(path):
            os.mkdir(path)
        #end if
    #end for
    
    # let's go !
    dataset = Dataset()
    trainer = Trainer(dataset)
    
    for epoch in range(N_EPOCHS):
        print("Epoch:", epoch + 1, "/", N_EPOCHS)
        
        trainer.train_one_epoch(epoch)
        
        print()
    #end for
    
    print("Training complete!")
    print()
#end with


if __name__ == "__main__":
    main()
#end if
