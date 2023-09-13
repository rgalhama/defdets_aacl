# coding:utf-8
"""
Name : train_bert_incremental.py
Author : Raquel G. Alhama
Desc:
"""
import os, sys, inspect
import argparse
import torch
from pathlib import Path
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast, BertTokenizer
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
SCRIPT_FOLDER = os.path.realpath(os.path.abspath(
    os.path.split(inspect.getfile(inspect.currentframe()))[0]))
PROJECT_FOLDER = os.path.join(SCRIPT_FOLDER, os.pardir)
DATA_PATH = os.path.join(PROJECT_FOLDER, "data")
sys.path.insert(0, PROJECT_FOLDER)
#from src.data.LDPBertReader import LDPDataset


print("Cuda available:", torch.cuda.is_available())

def create_model(config):
    model = BertForMaskedLM(config=config)
    print("Number of parameters: ", model.num_parameters())
    return model


def prepare_data(datafile, tokenizer, session):

    #alternative:
    #ldp=LDPDataset(tokenizer)

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=datafile,
        block_size=64
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    return dataset, data_collator


def train_bert_session(model, tokenizer, session, datafile, name, modelconfig, trainingconfig, outputpath, tokenizerpath):

    #Prepare data
    print("Preparing data for session {}...".format(session))
    dataset, data_collator = prepare_data(datafile, tokenizer, session)

    #Prepare trainer
    trainer = Trainer(model=model,
                      args=trainingconfig,
                      data_collator=data_collator,
                      train_dataset=dataset
                     )

    #Start training
    train_output=trainer.train()

    #(the train loss and other metrics can be inspected from train_output)

    #Save model checkpoint
    trainer.save_model(os.path.join(outputpath, name))


def train_bert_incremental(modelconfig, trainingconfig, path_to_data, output_path, tokenizerpath, max_phase,
                           input_fnames="", output_fnames=""):

    if input_fnames is None or input_fnames=="":
        input_fnames= "parent_utterances_session_{}.csv"


    #Prepare tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(tokenizerpath, local_files_only=True, max_len=128)
    modelconfig.vocab_size=tokenizer.vocab_size
    model=create_model(modelconfig)

    for session in range(1,max_phase+1):

        print("Training for session/phase {}:".format(session))
        datafile = os.path.join(path_to_data, input_fnames.format(session))

        train_bert_session(model, tokenizer, session, datafile, output_fnames.format(session), modelconfig,
                           trainingconfig, output_path, tokenizerpath)

if __name__ == "__main__":

    if len(sys.argv) > 1:

        parser = argparse.ArgumentParser()
        parser.add_argument("--path_to_data", type=str)
        parser.add_argument("--outputpath", type=str)
        parser.add_argument("--tokenizerpath", type=str)
        parser.add_argument("--input_fnames", type=str, default="")
        parser.add_argument("--output_fnames", type=str, default="")
        parser.add_argument("--max_phase", type=int, default=12, help="Last phase or session.")
        args = parser.parse_args()

        path_to_data = args.path_to_data
        outputpath = args.outputpath
        tokenizerpath=args.tokenizerpath
        output_fnames=args.output_fnames
        input_fnames=args.input_fnames
        max_phase=args.max_phase
        print(args)

    else:
        #default args, for quick testing
        print("No arguments provided. Running default configuration...")
        path_to_data = os.path.join(DATA_PATH, 'processed/LDP_full/sessions/')
        outputpath = os.path.join(DATA_PATH, "trained_models/BERT_over_full_LDP")
        tokenizerpath=os.path.join(DATA_PATH, "trained_models/BERT_over_full_LDP/WordPieceTokenizer/")
        output_fnames="BERT-LDP-incrsession-{}"
        input_fnames=""
        max_phase=12

    #Bert Configuration
    modelconfig = BertConfig(
        seed=118999,
        num_attention_heads=2,
        num_hidden_layers=1,
        type_vocab_size=1
    )

    #Trainer Configuration
    trainingconfig = TrainingArguments(
        output_dir=outputpath,
        overwrite_output_dir=True,
        per_gpu_train_batch_size=64,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        resume_from_checkpoint=None
    )

    #Training with incremental data
    train_bert_incremental(modelconfig, trainingconfig, path_to_data, outputpath, tokenizerpath,
                           max_phase, input_fnames=input_fnames, output_fnames=output_fnames)

