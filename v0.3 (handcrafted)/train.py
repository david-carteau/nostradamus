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
## AUTHOR: David Carteau, France, August 2025                               ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## PURPOSE:                                                                 ##
## Train the neural network (input: fenstring, output: best move to play)   ##
##############################################################################

##############################################################################
## This work was largely inspired by Andrej Karpathy's excellent tutorial:  ##
##               https://www.youtube.com/watch?v=kCc8FmEb1nY                ##
##                   https://github.com/karpathy/nanoGPT                    ##
##                                                                          ##
## A big thank you to Andrej for sharing this invaluable material!          ##
## David.                                                                   ##
##############################################################################

import os
import gzip
import torch
import pickle
import random

from tqdm import tqdm


##############################################################################
## VARIABLES SECTION (adjust if needed)                                     ##
##############################################################################

# network architecture
N_FEATURES = 256
N_HEADS  = 8
N_LAYERS = 8

# whether or not to use a CLS token
# (see here for more information: https://h2o.ai/wiki/classify-token/)

# set to True to add a special 65th token, which will aggregate the information
# from the entire position (which is encoded with 64 tokens, one per square)

# set to False if you want to use all 64 tokens; however, this will require a
# lot more parameters, resulting in increased memory usage and slower training
USE_CLS_TOKEN = True

# learning rate scheduler
LR = [0.0005] * 3

# batch size
# if you change this value and encountered problems,
# remove the./cache folder and/or delete all of its content
BATCH_SIZE = 2 * 1024

# weight decay, i.e. try to reduce the magnitude of weights and biases
DECAY = 1e-5

# random seed
# 21.05.2014 = date of Orion's first public release :-)
SEED = 21052014

# device used for training
# set to "auto" for automatic selection
# set to "cuda" (for Nvidia GPUs), "mps" (for Apple Silicon), etc.
TRAINING_DEVICE = "auto"


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
        
        print("Done!")
        print()
        
        self.batches = [i for i in range(batch)]
    #end def
    
    def __len__(self):
        return len(self.batches)
    #end def
    
    def __iter__(self):
        self.batch = -1
        return self
    #end def
    
    def __next__(self):
        self.batch += 1
        
        if self.batch == len(self.batches):
            random.shuffle(self.batches)
            raise StopIteration
        #end if
        
        batch = self.batches[self.batch]
        
        return self.load_batch(batch)
    #end def
    
    # get board and move from each line of ./positions/positions-shuffled.txt
    def get_sample(self, line):
        fen, stm, cas, enp, move = line.strip().split()
        
        rows = fen.split("/")
        
        assert len(rows) == 8
        
        square = 0
        
        pieces = "pnbrqk"
        digits = "12345678"
        
        if stm == "w":
            stm_chars = pieces.upper()
        else:
            stm_chars = pieces
        #end if
        
        board = [0] * 64
        
        for row in rows[::-1]:
            for char in row:
                if char in digits:
                    square += digits.find(char)
                else:
                    if stm == "w":
                        sq = square
                    else:
                        sq = square ^ 56
                    #end if
                    
                    idx = pieces.find(char.lower())
                    
                    if char in stm_chars:
                        board[sq] = (2 * idx) + 1 # odd number
                    else:
                        board[sq] = (2 * idx) + 2 # even number
                    #end if
                #end if
                square += 1
            #end for
        #end for
        
        assert square == 64
        
        fr = "abcdefgh".index(move[0]) + 8 * "12345678".index(move[1])
        to = "abcdefgh".index(move[2]) + 8 * "12345678".index(move[3])
        
        if stm == "b":
            fr ^= 56
            to ^= 56
        #end if
        
        move = (fr, to)
        
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


# single-head attention
class Attention(torch.nn.Module):
    def __init__(self, C, H):
        super().__init__()
        
        self.key   = torch.nn.Linear(C, H, bias=False)
        self.query = torch.nn.Linear(C, H, bias=False)
        self.value = torch.nn.Linear(C, H, bias=False)
    #end def
    
    def forward(self, x):
        k = self.key(x)   # (B, T, H)
        q = self.query(x) # (B, T, H)
        v = self.value(x) # (B, T, H)
        
        H = k.shape[-1]
        
        w = q @ k.transpose(-2, -1) * H**-0.5 # (B, T, H) @ (B, H, T) --> (B, T, T)
        w = torch.softmax(w, dim=-1)
        
        return w @ v # (B, T, T) @ (B, T, H) --> (B, T, H)
    #end def
#end class


# basic (i.e. not batched) multi-head attention
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_features, n_heads):
        super().__init__()
        
        C = n_features
        H = n_features // n_heads
        
        self.heads = [Attention(C, H) for _ in range(n_heads)]
        self.heads = torch.nn.ModuleList(self.heads)
    #end def
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
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
            self.ffd = torch.nn.Sequential(
                torch.nn.Linear(n_features, 4 * n_features),
                torch.nn.ReLU(),
                torch.nn.Linear(4 * n_features, n_features),
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
        
        if USE_CLS_TOKEN:
            self.pieces_emb  = torch.nn.Embedding(14, N_FEATURES)
            self.squares_emb = torch.nn.Embedding(65, N_FEATURES)
        else:
            self.pieces_emb  = torch.nn.Embedding(13, N_FEATURES)
            self.squares_emb = torch.nn.Embedding(64, N_FEATURES)
        #end if
        
        layers = []
        
        for _ in range(N_LAYERS):
            layer = TransformerBlock(n_features=N_FEATURES, n_heads=N_HEADS)
            layers.append(layer)
        #end for
        
        self.layers = torch.nn.ModuleList(layers)
        
        if USE_CLS_TOKEN:
            self.head = torch.nn.Linear(N_FEATURES, 64)
        else:
            self.head = torch.nn.Linear(64*N_FEATURES, 64)
        #end if
    #end def
    
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    #end def
    
    def forward(self, board):
        if USE_CLS_TOKEN:
            batch_size = board.shape[0]
            cls_token  = torch.tensor([13] * batch_size, device=board.device).view(-1, 1)
            board_with_cls_token = torch.cat((board, cls_token), dim=1)
            
            x = self.pieces_emb(board_with_cls_token) + self.squares_emb(torch.arange(65, device=board.device))
        else:
            x = self.pieces_emb(board) + self.squares_emb(torch.arange(64, device=board.device))
        #end if
        
        for layer in self.layers:
            x = layer(x)
        #end for
        
        if USE_CLS_TOKEN:
            x = x[:,-1,:]
        else:
            x = x.view(board.shape[0], -1)
        #end if
        
        x = self.head(x)
        
        # we can help the model!
        
        # the 'fr' squares empty or with an opponent piece are neutralised
        fr = torch.where((board % 2) == 0, float("-inf"), x)
        
        # the 'to' squares with a piece from the player are neutralised
        to = torch.where((board % 2) == 1, float("-inf"), x)
        
        return fr, to
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
        
        y_true_fr = y_true[:,0]
        y_true_to = y_true[:,1]
        
        # re-init gradients
        self.optimizer.zero_grad()
        
        # predict output (i.e. forward pass)
        y_pred_fr, y_pred_to = self.model(X_true)
        
        # compute loss
        loss = self.criterion(y_pred_fr, y_true_fr) + self.criterion(y_pred_to, y_true_to)
        
        # clip gradients by value (optional)
        #torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
        
        # clip gradients by norm (alternative, optional)
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
        
        # back-propagate the loss
        loss.backward()
        
        # update model weights
        self.optimizer.step()
        
        # clip model weights (in the perspective of quantization)
        with torch.no_grad():
            for param in self.model.parameters():
                param.clamp_(-Q, Q)
            #end for
        #end with
        
        return loss.item()
    #end def
    
    def train_one_epoch(self, epoch):
        loss = 0
        losses = []
        
        # iterate through the dataset
        for X, y in (pbar := tqdm(self.dataset)):
            loss = self.train_one_batch(X, y)
            losses.append(loss)
            
            loss = losses[-32:]
            loss = sum(loss) / len(loss)
            
            last = self.scheduler.get_last_lr()[0]
            
            pbar.set_postfix({"LR:": format(last), "Loss:": format(loss)})
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
    print(f'Nostradamus 0.3 (2025)')
    print("https://github.com/david-carteau")
    print()
    
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
