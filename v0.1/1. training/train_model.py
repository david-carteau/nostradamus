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
## NAME: train_model.py                                                     ##
## AUTHOR: David Carteau, France, October 2024                              ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## IMPORTANT !                                                              ##
## This chess engine is very weak: its only purpose is to see how language  ##
## models can be trained to play chess ;-)                                  ##
##############################################################################


import os
import torch
import pickle

from tqdm import tqdm
from utils import transform
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizerFast
from transformers import PhiConfig, PhiForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

MAX_LENGTH = 82
N_SAMPLES_LIMIT = 32*1024*1024


class SamplesDataset(Dataset):
    def __init__(self, tokenizer):
        if not os.path.exists("./samples.pickle"):
            self.convert(tokenizer)
        #end if
        
        with open("./samples.pickle", "rb") as file:
            self.samples = pickle.load(file)
        #end with
        
        print(len(self.samples), "games")
    #end def
    
    def __len__(self):
        return len(self.samples)
    #end def
    
    def __getitem__(self, i):
        return torch.tensor(self.samples[i])
    #end def
    
    def convert(self, tokenizer):
        samples = []
        
        with open(f'./samples.fen', "rt", encoding="utf-8") as file:
            for line in tqdm(file):
                position = transform(line)
                
                ids = tokenizer.encode(position)
                
                assert len(ids) == MAX_LENGTH
                
                samples += [ids]
                
                if N_SAMPLES_LIMIT is None:
                    continue
                #end if
                
                if len(samples) >= N_SAMPLES_LIMIT:
                    break
                #end if
            #end for
        #end with
        
        with open(f'./samples.pickle', "wb") as file:
            pickle.dump(samples, file)
        #end with
        
        samples = None
    #end def
#end def


def main():
    tokenizer = PreTrainedTokenizerFast.from_pretrained("./model")
    
    bos_token_id = tokenizer.encode("<s>")[0]
    eos_token_id = tokenizer.encode("</s>")[0]
    
    hidden_size = 512
    
    config = PhiConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=MAX_LENGTH,
        hidden_size=hidden_size,
        intermediate_size=(hidden_size * 4),
        num_hidden_layers=4,
        num_attention_heads=32,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id
    )
    
    dataset = SamplesDataset(tokenizer)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    for epoch in range(2):
        print("epoch", epoch)
        
        path = f'./model-{epoch}'
        path_model = f'./model-{epoch}/model.safetensors'
        
        if os.path.exists(path) and os.path.exists(path_model):
            continue
        #end if
        
        if epoch == 0:
            model = PhiForCausalLM(config=config)
        else:
            model = PhiForCausalLM.from_pretrained(f'./model-{epoch-1}')
        #end if
        
        print(model.num_parameters(), "parameters")
        
        training_args = TrainingArguments(
            output_dir=path,
            overwrite_output_dir=True,
            log_level="info",
            logging_strategy="steps",
            logging_steps=10,
            dataloader_drop_last=True,
            learning_rate=0.0005,
            lr_scheduler_type="constant",
            weight_decay=0.1,
            num_train_epochs=1,
            per_device_train_batch_size=256,
            #gradient_accumulation_steps=1,
            save_strategy="epoch",
            fp16=True,
            seed=epoch
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        
        trainer.train()
        
        trainer.save_model(path)
        
        print("--------")
        print()
    #end for
#end def


if __name__ == "__main__":
    main()
#end if
