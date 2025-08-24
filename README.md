## The Nostradamus UCI Chess Engine

![Logo](https://github.com/david-carteau/nostradamus/blob/main/v0.1%20(decoder)/nostradamus.jpg)

The **Nostradamus** UCI chess engine uses **language models** inspired techniques to play chess:
- unlike other engines, it does not rely on a traditional search tree to find the optimal sequence of moves
- instead, it uses a network architecture based on [Transformers](https://arxiv.org/abs/1706.03762) to predict the next best move given a specific position

The engine is still in the **early stages of development** and is currently very weak (see below). **Do not consider using it**, unless you are curious ;-)

<br/>

## Performance

| Rank | Name            | Elo  | +/- | Games | Score | Draw |
| ---: | :-------------- | ---: | --: | ----: | ----: | ----: |
| 1    | Nostradamus 0.3 | 352  | 24  | 1200  | 88.4% | 16.3%|
| 2    | Nostradamus 0.2 | 207  | 19  | 1200  | 76.8% | 25.8%|
| 3    | Nostradamus 0.1 | 108  | 17  | 1200  | 65.0% | 30.5%|
| 4    | Capture 1.0     | -57  | 17  | 1200  | 41.9% | 25.2%|
| 5    | POS 1.20        | -105 | 17  | 1200  | 35.3% | 28.2%|
| 6    | Cerebrum 1.0    | -153 | 15  | 1200  | 29.3% | 43.5%|
| 7    | Random 1.0      | -324 | 19  | 1200  | 13.4% | 26.7%|

<br/>

Testing conditions:
- _Hert500.pgn_ book, 100 repeated games by engine, depth = 1
- Capture 1.0 and Random 1.0 were specifically developed to evaluate the Nostradamus engine
- POS is ranked 417 Elo on [CCRL 40/2 Archive](https://www.computerchess.org.uk/ccrl/402.archive/)

<br/>

## How it works

I'm fascinated by the performance of small language models such as [Microsoft Phi 3.5](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) and have decided to train one to play chess ;-)

Given a sentence, language models try to predict the most likely word to follow. Actually, they do not work with words, but with "tokens", which can be seen as "subwords".

I first trained a model with "sentences" consisting of simple sequences of moves, e.g. `e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 a7a6 f1d3`, with the aim of predicting the next best move, e.g. `g8f6`.

This is very similar to the way language models work. The advantage is that you can ask the model to predict the following moves and get a full principal variation !

However, with this approach it's very difficult for the model to know the position of the pieces: it has to follow every piece from the beginning of the game (e.g. after `... g1f3 ... f3d4` : there is now a knight on `d4`). This leads to a lot of illegal moves when trying to predict the next best move.

Instead of the sequence of moves, I decided to give the (textual) representation of the board (as a fenstring) as the "sentence": this drastically reduces the generation of illegal moves !

This gave me **v0.1**, based on the [Microsoft Phi 3.5](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) model architecture (decoder-only architecture).

I then realised that given a position, trying to predict the best move to play could be seen as... a translation problem !

I switched to such a specialised model (encoder-decoder architecture), [Google T5](https://huggingface.co/google-t5/t5-base), which gave me the **v0.2** and a nice improvement in strength !

I was then interested in delving into the [Transformers](https://arxiv.org/abs/1706.03762) architecture, and implemented a first (and very simple) custom network in **v0.3** (encoder-only architecture).

<br/>

## Installation

In order to use the UCI engine (v0.1, v0.2 or v0.3), you will need the following:
- a Python runtime environment: https://www.python.org/
- for v0.1 and v0.2: some Python libraries (see `./data preparation/1. install_libraries.bat`)
- for v0.3: some Python libraries (see `./install/requirements.txt`)
- to download the [package](https://www.orionchess.com/download/Nostradamus-v0.1-to-v0.3.zip) containing the engines and their respective models

<br/>

Note:
- there's no need for a powerful GPU for inference!
- i.e. if you only want to use the Nostradamus engine and not to train models, you can install the CPU version of PyTorch

<br/>

## Data used for the training

Models were trained using data:
- extracted from [CCRL](https://www.computerchess.org.uk/ccrl/) games (CCRL 40/2 Archive, CCRL Blitz and CCRL 40/15)
- excluding drawn games (i.e. `1/2-1/2` result)

<br/>

This resulted in a dataset:
- of ~81M positions for v0.1 and v0.2 (as of December 2024)
- of ~86M positions for v0.3 (as of August 2025)

<br/>

## If you want to train your own models

In v0.1 and v0.2, source code is provided to:

- train the tokenizer, responsible for converting fenstrings and moves into sequences of "tokens"
- train the language model, responsible for predicting the most likely move for a given fenstring

<br/>

In v0.3, source code is also provided to:

- train the model responsible for predicting the most likely move for a given fenstring

<br/>

For v0.1 and v0.2, you'll need:

- an **NVIDIA GPU** (do not consider training on the CPU, as it would be very slow)
- the `pgn-extract` tool (https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/)
- a `games.pgn` file containing games in PGN-format, placed in a `pgn` folder (to be created)

<br/>

For v0.3:

- training has been also successfully performed on an **Apple Silicon CPU** (Mac mini M4 Pro 10/16, 24 GB RAM)
- you can provide several PGN files (which will be automatically combined) in the `./preparation/pgn` folder

<br/>

Some useful figures:

- for v0.1 and v0.2, consider 5-6 hours per epoch on an Nvidia RTX 4070 Ti (12GB VRAM GPU)
- for v0.3, consider 4-5 hours per epoch on the same GPU (much slower on the Mac mini)

<br/>

## Contribute to the experiment!

If you would like to contribute, please contact me via the [talkchess.com](https://www.talkchess.com) forum!

<br/>

Next steps might include:

- enhancing the v0.3 handcrafted model (the current approach is too basic)
- playing games with stronger opponents to get more unbalanced positions

<br/>


## Copyright, licence

Copyright 2024-2025 by David Carteau. All rights reserved.

The Nostradamus UCI chess engine is licensed under the **MIT License** (see "LICENSE" and "license.txt" files).
