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
## AUTHOR: David Carteau, France, November 2024                             ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## IMPORTANT !                                                              ##
## This chess engine is very weak: its only purpose is to see how language  ##
## models can be trained to play chess ;-)                                  ##
##############################################################################


import os
import torch

from tqdm import tqdm
from utils import transform
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizerFast
from transformers import T5Config, T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

MAX_LENGTH = 82


class SamplesDataset(Dataset):
    def __init__(self, tokenizer):
        self.samples = []
        self.tokenizer = tokenizer
        
        with open(f'./samples.fen', "rt", encoding="utf-8") as file:
            for line in tqdm(file):
                self.samples.append(line.strip())
            #end for
        #end with
    #end def
    
    def __len__(self):
        return len(self.samples)
    #end def
    
    def __getitem__(self, i):
        sample = self.samples[i]
        
        position, move = transform(sample)
        
        sample = self.tokenizer(text=position, text_target=move, max_length=MAX_LENGTH, truncation=True)
        
        sample = {
            "input_ids": sample["input_ids"],
            "labels"   : sample["labels"]
        }
        
        return sample
    #end def
#end def


def main():
    tokenizer = PreTrainedTokenizerFast.from_pretrained("./model")
    
    pad_token_id = tokenizer.encode("<pad>")[0]
    eos_token_id = tokenizer.encode("</s>")[0]
    
    config = T5Config(
        vocab_size=len(tokenizer),
        d_model=256,          #512 by default
        d_kv=32,              #64 by default
        d_ff=1024,            #2048 by default
        num_layers=4,         #6 by default
        num_decoder_layers=4, #num_layers by default
        num_heads=8,          #8 by default
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        decoder_start_token_id=pad_token_id
    )
    
    print("Reading dataset...")
    
    dataset = SamplesDataset(tokenizer)
    
    print()
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    
    for epoch in range(1):
        print("epoch", epoch)
        
        path = f'./model-{epoch}'
        path_model = f'./model-{epoch}/model.safetensors'
        
        if os.path.exists(path) and os.path.exists(path_model):
            continue
        #end if
        
        if epoch == 0:
            model = T5ForConditionalGeneration(config=config)
        else:
            model = T5ForConditionalGeneration.from_pretrained(f'./model-{epoch-1}')
        #end if
        
        print(model.num_parameters(), "parameters")
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=path,
            overwrite_output_dir=True,
            log_level="info",
            logging_strategy="steps",
            logging_steps=32,
            dataloader_drop_last=True,
            dataloader_num_workers=2,
            learning_rate=0.0005,
            #lr_scheduler_type="constant",
            weight_decay=0.1,
            num_train_epochs=1,
            per_device_train_batch_size=1024,
            #gradient_accumulation_steps=1,
            save_strategy="epoch",
            fp16=True,
            seed=epoch
        )
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
            processing_class=tokenizer
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