## The Nostradamus UCI Chess Engine

![Logo](https://github.com/david-carteau/nostradamus/blob/main/v0.1%20(decoder)/nostradamus.jpg)

The **Nostradamus** UCI chess engine uses a **small language model** to play chess:
- unlike other engines, it doesn't rely on a traditional search tree to find the best combination of moves
- it uses instead a language model to predict the best move given a specific position

The engine is at an **early stage of development** and is currently very weak (somewhere around 700 elo, see below). **Do not consider using it**, except if you are curious ;-)

<br/>

## Performance

Using _Hert500.pgn_ book:

| Rank | Name            | Elo  | +/- | Games | Score |  Draw |
| ---: | :-------------- | ---: | --: | ----: | ----: | ----: |
|    1 | Nostradamus 0.2 |  309 |  22 |  1000 | 85.5% | 23.3% |
|    2 | Nostradamus 0.1 |  188 |  20 |  1000 | 74.7% | 27.8% |
|    3 | Capture 1.0     |   -8 |  18 |  1000 | 48.9% | 27.5% |
|    4 | POS 1.20        |  -67 |  18 |  1000 | 40.5% | 34.1% |
|    5 | Cerebrum 1.0    | -106 |  15 |  1000 | 35.1% | 50.9% |
|    6 | Random 1.0      | -297 |  19 |  1000 | 15.3% | 30.6% |

<br/>

Using _DC-UltraFair.pgn_ book:

| Rank | Name            | Elo  | +/- | Games | Score |  Draw |
| ---: | :-------------- | ---: | --: | ----: | ----: | ----: |
|    1 | Nostradamus 0.2 |  292 |  22 |  1000 | 84.3% | 23.8% |
|    2 | Nostradamus 0.1 |  208 |  19 |  1000 | 76.8% | 31.6% |
|    3 | Capture 1.0     |  -13 |  18 |  1000 | 48.2% | 28.4% |
|    4 | POS 1.20        |  -64 |  18 |  1000 | 40.9% | 32.0% |
|    5 | Cerebrum 1.0    | -108 |  14 |  1000 | 34.9% | 53.6% |
|    6 | Random 1.0      | -303 |  20 |  1000 | 14.9% | 29.2% |

_Note: Capture 1.0 and Random 1.0 are engines that have been specifically developed to assist in the evaluation of the Nostradamus engine._

<br/>

## How it works

I'm fascinated by the performance of small language models like [Microsoft Phi 3.5](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) and decided to train one to play chess ;-)

Given a sentence, language models try to predict the most likely word to follow. Actually, they do not work with words, but with "tokens", which can be seen as "subwords".

I first trained a model with "sentences" consisting of simple sequences of moves, e.g. `e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 a7a6 f1d3`, with the aim of predicting the next best move, e.g. `g8f6`.

This is very similar to the way language models work. The advantage is that you can ask the model to predict the following moves and get a full principal variation!

However, with this approach it's very difficult for the model to know the position of the pieces: it has to follow every piece from the beginning of the game (e.g. after `... g1f3 ... f3d4` : there is now a knight on `d4`). This leads to a lot of illegal moves when trying to predict the next best move.

Instead of the sequence of moves, I decided to give the (textual) representation of the board (as a fenstring) as a "sentence": this drastically reduces the generation of illegal moves!

This gave me **v0.1**, based on the [Microsoft Phi 3.5](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) model architecture (decoder-only architecture).

I then realised that given a position, trying to predict the best move to play could be seen as... a translation problem !

I switched to such a specialised model, [Google T5](https://huggingface.co/google-t5/t5-base), which gave me the **v0.2** and a nice improvement in strength !

<br/>

## Installation

To use the UCI engine (v0.1 or v0.2), you'll need:
- a Python runtime: https://www.python.org/
- some Python libraries (see `./data preparation/1. install_libraries.bat`)
- to [download](https://www.orionchess.com/download/Nostradamus-v0.1-to-v0.2.zip) the package containing the engines and their respective models

<br/>

## Data used for the training

For v0.1 and v0.2, the models were trained on the same set of 81M positions:
- extracted from [CCRL](https://www.computerchess.org.uk/ccrl/) games
- only keeping non-adjudicated games
- excluding drawn games (i.e. `1/2-1/2` result)

<br/>

## If you want to train your own language models

Source code is provided to:

- train the tokeniser, responsible for converting fenstrings and moves into sequences of "tokens"
- train the language model, responsible for predicting the most likely move for a given fenstring

<br/>

To use both trainers, you'll need:

- a **NVIDIA GPU** (do not consider training on the CPU, as it would be very slow)
- a **Python** runtime: https://www.python.org/
- some Python libraries (see `./data preparation/1. install_libraries.bat`)
- the `pgn-extract` tool (https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/)
- a `games.pgn` file containing games in PGN format, put in a `pgn` folder (to create)

<br/>

Some useful figures:

- provided models were trained targeting a 12GB VRAM GPU (Nvidia RTX 4070 Ti)
- each version needed 5 to 6 hours of training
- a computer with at least 64 BG of RAM is strongly recommended !

<br/>

## Contribute to the experiment !

If you want to contribute, do not hesitate to reach me through the [talkchess.com](https://www.talkchess.com) forum !

<br/>

Next steps might include:
- using more positions/epochs to train the language models
- using a handcrafted model
- mix fenstring and sequence of moves as inputs

<br/>

## Copyright, licence

Copyright 2024 by David Carteau. All rights reserved.

The Nostradamus UCI chess engine is licensed under the **MIT License** (see "LICENSE" and "license.txt" files).
