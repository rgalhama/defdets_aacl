# coding:utf-8
"""
Name : train_tokenizer.py.py
Author : Raquel G. Alhama
Desc: Trains BWP tokenizer.
"""
import os, sys, inspect
from tokenizers.implementations import ByteLevelBPETokenizer, BertWordPieceTokenizer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
SCRIPT_FOLDER = os.path.realpath(os.path.abspath(
    os.path.split(inspect.getfile(inspect.currentframe()))[0]))
PROJECT_FOLDER = os.path.join(SCRIPT_FOLDER, os.pardir)
sys.path.insert(0, PROJECT_FOLDER)

def create_WordPiecetokenizer(pth):

    # Initialize a tokenizer
    tokenizer = BertWordPieceTokenizer()
    tokenizer.pre_tokenizer = Whitespace() #pretokenization (split into words)

    # Customize training
    tokenizer.train(files=[pth], vocab_size=30_000, min_frequency=2, special_tokens=[
        "<s>",
        "[UNK]",
        "[PAD]",
        "[CLS]",
        "</s>",
        "[MASK]",
        "[SEP]",
    ])

    return tokenizer

if __name__ == "__main__":

    datapath=os.path.join(PROJECT_FOLDER, "data")

    # Args
    # 1: path to input data
    # 2: path to output folder
    if len(sys.argv) <=1:
        print("Arguments not provided: using script with default parameters (LDP).")
        ldppath=datapath+ "/processed/LDP_full/ldp_parents.txt"
        outputfolder=os.path.join(datapath, 'trained_models/BERT_over_full_LDP/WordPieceTokenizer')
    else:
        ldppath=sys.argv[1]
        outputfolder=sys.argv[2]

    if not os.path.exists(ldppath):
        raise Exception("{} does not exist".format(ldppath))

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    #Create tokenizer
    tokenizer=create_WordPiecetokenizer(ldppath)

    #Output
    tokenizer.save_model(outputfolder)
    tokenizer.save(os.path.join(outputfolder, "config.json"), True)
    # tpr=PreTrainedModel(tokenizer)
    # tpr.save_pretrained(os.path.join(datapath, "trained_models/tokenizer"), save_config=True)

    print("Done. Results at:")
    print(outputfolder)