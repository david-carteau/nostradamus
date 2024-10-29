## The Nostradamus UCI Chess Engine

![Logo](/v0.1/2. engine/nostradamus.jpg)

The **Nostradamus** UCI chess engine uses a **small language model** which is specially trained to predict the best move given a particular position.

I'm fascinated by the performance of small language models like [Microsoft Phi 3.5](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) and decided to train one to play chess.

The engine is at an early stage of development and is currently very weak (it scores about 80% against a random mover).

Do not consider using it, except if you are curious ;-)

Source code is available to:

    **Train the tokeniser**, responsible for converting fenstrings into sequences of tokens
    **Train the language model**, responsible for predicting the most likely move for a given fenstring
    **Use the language model in a chess engine compatible with the UCI protocol.

To use both trainers you'll need

- an NVIDIA GPU (do not consider training on the CPU, as it would be very slow)
- a **Python** runtime: https://www.python.org/
- some Python libraries: see '''0. install_libraries.bat''' script (adapt it to your platform/environment)
- the [pgn-extract] tool (https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/)

To use the UCI engine, use the packaged version (target: Windows 11 + Python 3.11).

If you don't want to contribute, you can reach me at [talkchess.com](https://www.talkchess.com) 🌟.

## Copyright, licence

Copyright 2024 by David Carteau. All rights reserved.

The Nostradamus UCI chess engine is licensed under the **MIT License** (see "LICENSE" and "license.txt" files).