## The Nostradamus UCI Chess Engine

![Logo](/v0.1/nostradamus.jpg)

The **Nostradamus** UCI chess engine uses a **small language model** which is specially trained to predict the best move given a particular position.

The engine is at an **early stage of development** and is currently very weak (it scores about 80% against a random mover).

**Do not consider using it**, except if you are curious ;-)

<br/>

## How it works

I'm fascinated by the performance of small language models like [Microsoft Phi 3.5](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) and decided to train one to play chess ;-)

I first tried to train a model with sequences of moves, e.g. `e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 a7a6 f1d3`, with the goal of predicting the next best move, e.g. `g8f6`. This is how language models work: they try to complete a sentence with the next most likely word, and so on.

The advantage is that you can ask the model to predict the following moves and get a full principal variation. The disadvantage is that it is very difficult for the model to know which pieces are where, it has to follow every piece from the beginning of the game (e.g. after `... g1f3 ... f3d4` : there's now a knight on d4). This leads to a lot of illegal moves when trying to predict the next best move.

I then helped the model by not giving the sequence of moves, but the representation of the board (as a fenstring) : it dramatically reduces the generation of illegal moves, and performs slightly better against a random mover, event with a limited amount of data.

<br/>

## Installation

To use the UCI engine, you'll need:
- a **Python** runtime: https://www.python.org/
- some **Python libraries**: `pip install --user chess==1.11.1 torch==2.4.1 transformers==4.46.0`
- to train a language model (see below)

<br/>

Or directly [download](https://www.orionchess.com/download/Nostradamus-v0.1.zip) the engine packaged with:
- a pre-trained language model (see below for more details on the data used)
- Python libraries for Windows 11 + Python 3.11

<br/>

## Language model training

Source code is provided to:

- Train the tokeniser, responsible for converting fenstrings into sequences of tokens
- Train the language model, responsible for predicting the most likely move for a given fenstring
- Use the language model in a chess engine compatible with the UCI protocol

<br/>

To use both trainers, you'll need:

- a **NVIDIA GPU** (do not consider training on the CPU, as it would be very slow)
- a **Python** runtime: https://www.python.org/
- some **Python libraries**: see `0. install_libraries.bat` script (adapt it to your platform/environment)
- the `pgn-extract` tool (https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/)
- a `games.pgn` file containing games in PGN format, put in a `pgn` folder (to create)

<br/>

In its 0.1 version, the model has been trained:
- using 32M positions extracted from [CCRL](https://www.computerchess.org.uk/ccrl/) games
- excluding drawn games (i.e. `1/2-1/2` result)
- targeting a 6GB VRAM GPU

<br/>

## Contribute to the experiment!

If you want to contribute, you can reach me at [talkchess.com](https://www.talkchess.com) !

<br/>

Next steps might include:
- using more positions/epochs to train the language model
- using an encoder/decoder model
- using a handcrafted model
- mix fenstring and sequence of moves as inputs

<br/>

## Copyright, licence

Copyright 2024 by David Carteau. All rights reserved.

The Nostradamus UCI chess engine is licensed under the **MIT License** (see "LICENSE" and "license.txt" files).
