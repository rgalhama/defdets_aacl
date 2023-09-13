# coding:utf-8
"""
Name : test_masked_det_incr.py
Author : Raquel G. Alhama
Desc:
    Test BERT model on masked sentences.
    This version tests sentences by different subjects separately.

    Input: trained model, masked sentences
    Output: probabilities of items for masked position (+ derived metrics such as ranking, preferred det, etc)
"""

import sys, os, inspect, argparse
import torch
import pandas as pd
from transformers import pipeline
from transformers import BertTokenizerFast
SCRIPT_FOLDER = os.path.realpath(os.path.abspath(
    os.path.split(inspect.getfile(inspect.currentframe()))[0]))
PROJECT_FOLDER = os.path.join(SCRIPT_FOLDER, os.pardir, os.pardir)
sys.path.insert(0, PROJECT_FOLDER)
from argparse import Namespace
from src.data.ldpdata import LDPChildren, LDPParents
from src.data.manchesterdata import ManchesterChildren


def select_nouns_w_2_dets(ifile, ofile, det_col):
    df=pd.read_csv(ifile, sep=";", error_bad_lines=False, engine="python") #corrected this because of error in session 6
    df = df[df.preferred_det.notnull()]

    if len(df) == 0:
        return None

    fdf=None

    for subject in df.subject.unique():
        sdf=df[df.subject == subject][['session', 'h_noun', det_col]]

        #Group all the combinations and count them
        cdf=sdf.groupby(['h_noun', det_col, 'session']).size().to_frame('counts')
        cdf=cdf.reset_index()

        #Select only nouns that have 2 preferred determinants
        cdf=cdf.groupby(['h_noun']).filter(lambda x: len(x) > 1)
        cdf['subject']=subject

        fdf=cdf if fdf is None else pd.concat([fdf,cdf])

    print("Saving nouns w 2 dets to...")
    fdf.to_csv(ofile, sep=";", index=False)
    print(ofile)

    return fdf


def test_session(dataset, session, who, modelname, modelpath, tokenizer, outputpath, downsampled_parents=False):

    assert(who in ('children', 'parents'))

    # 1. Test model with sentences (from children or parents) with masked determinants
    # Load data
    if dataset.lower() == "ldp": #note to self: there was an error here before (dataset.lower)
        p = LDPChildren() if who == 'children' else LDPParents()
    else:
        p = ManchesterChildren()

    samplesize=-1 #no downsampling

    if who=="parents" and downsampled_parents:
        c=LDPChildren()
        samplesize=c.get_number_utts_dets_session(session)
        del c

    utts = p.get_masked_dets(session, filter_dets=['a', 'an', 'the'], samplesize=samplesize)
    predsfile = outputpath + '{}_{}_testsession{}.csv'.format(modelname, who, session)

    # Test model
    test_model(utts, modelpath, tokenizer, predsfile)
    print("Test complete. Predictions at:")
    print(predsfile)

    # 2. Aggregate counts of a and the for each noun
    countsfile = outputpath + '{}_{}_testsession{}_nouns_w_2_dets.csv'.format(modelname, who, session)
    bertdf = select_nouns_w_2_dets(predsfile, countsfile, 'preferred_det')
    return bertdf

def get_entropy(probs):
    probabilities_tensor = torch.tensor(probs)
    entropy = -(probabilities_tensor * torch.log2(probabilities_tensor)).sum()
    return entropy
def get_preds(unmasker, masked_utt,topk=1,compute_entropy=True):
    preds = unmasker(masked_utt, top_k=tokenizer.vocab_size)
    probmass_topk = sum([pred['score'] for pred in preds[:topk]])
    entropy=None
    if compute_entropy:
        entropy=get_entropy([pred['score'] for pred in preds])
    prob_the, prob_a = 0, 0
    rank_the, rank_a = float('nan'), float('nan')
    rank = 0
    top_pred = preds[0]['token_str']
    for pred in preds:
        rank += 1
        if pred['token_str'] == 'the':
            prob_the = pred['score']
            rank_the = rank
        elif pred['token_str'] == 'a':
            prob_a = pred['score']
            rank_a = rank
    if prob_a == prob_the:
        pref = float('nan')
    else:
        pref = ['the', 'a'][prob_a > prob_the]
    return top_pred,pref,prob_the,prob_a,rank_the,rank_a,probmass_topk, entropy


def test_model(ldp_masked, model, tokenizer, ofile, topk=1):

    #Load Model
    unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)

    # Predict
    print('Testing %s ...'%model)
    with open(ofile, 'w') as fh:
        header='session;subject;h_det;h_noun;sentence;top_prediction;preferred_det;prob_the;prob_a;rank_the;rank_a;probmass_topk;entropy\n'
        fh.write(header)
        for session, subject, mu, chdet, noun in ldp_masked:
            top_pred, pref, prob_the, prob_a, rank_the, rank_a, probmass_topk, entropy = get_preds(unmasker, mu, topk=topk)
            fh.write("{};{};{};{};{};{};{};{:.3f};{:.3f};{};{};{};{}\n".format(
                session,subject,chdet,noun,mu,top_pred,pref,prob_the,prob_a,rank_the,rank_a,probmass_topk,entropy))

def start_test(args):
    for session in range(1, args.max_phase + 1):
        # Select which trained model to test (final, or incrmentally trained up to session)
        modelsession = args.max_phase if args.only_final_model else session
        modelname = args.modelname_template.format(modelsession)
        argums = (
            args.dataset,
            session,  # Session for test data
            args.who,
            modelname,
            os.path.join(args.path_to_models, modelname),
            tokenizer,
            args.outputpath
        )
        df = test_session(*argums, downsampled_parents=args.downsampled_parents)

def create_args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Manchester or LDP.")
    parser.add_argument("--who", type=str, help="Speaker of test utterances (children or parents).")
    parser.add_argument("--topk", type=int)
    parser.add_argument("--max_phase", type=int, default=12, help="Last phase or session.")
    parser.add_argument("--outputpath", type=str)
    parser.add_argument("--path_to_models", type=str)
    parser.add_argument("--tokenizerpath", type=str)
    parser.add_argument("--modelname_template", type=str, default="BERT-LDP-incrsession-{}")
    parser.add_argument("--only_final_model", action='store_true', help="Test only the last model "
                                                                        "(instead of incrementally trained model / session).")
    # parser.add_argument("--cumulative_data", action='store_true', help='Deprecated option; do not use.')
    args = parser.parse_args()

    return args

def create_args_default():

    # default args
    dataset = "ldp"
    who = "children" #"parents"
    downsampled_parents = False #True  # tested on sentences from parents, but downsampled to have as many as children's
    if downsampled_parents and who.lower() != "parents":
        raise Exception("Incompatible selection of arguments"(downsampled_parents))
    TOPK = 1
    training = "incremental"  # full or incremental
    outputpath = "../../results/results_masked_lm_id/a_the/{}/{}_training_top{}/" \
        .format(dataset.upper(), training, TOPK)
    if who.lower() == "parents":
        outputpath = outputpath.replace("{}_training_top{}".format(training, TOPK),
                                        "model_tested_on_parents{}_context".format("_downsampled" if downsampled_parents else ""))

    args = Namespace(
        topk=TOPK,
        TRAINING=training, \
        TYPEDET="a_the", \
        dataset=dataset, \
        who=who,
        downsampled_parents=downsampled_parents,
        max_phase=12,
        only_final_model=False, \
        modelname_template="BERT-LDP-incrsession-{}",  # "BERT-Manchester-incrphase-{}", #or "BERT_over_full_LDP"
        outputpath=outputpath, \
        tokenizerpath=PROJECT_FOLDER + "/data/trained_models/BERT_over_full_{}/WordPieceTokenizer/".format(dataset.upper()), \
        path_to_models=PROJECT_FOLDER + "/data/trained_models/")
    # cumulative_data=False

    return args

if __name__ == "__main__":
    device = torch.device("cuda")

    if len(sys.argv) > 1:
        create_args_parse()
    else:
        args=create_args_default()
    TOPK=args.topk

    #Create output directory if doesn't exist
    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    # Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizerpath, max_len=128)
    vocab_size=len(tokenizer.get_vocab())

    #Start test
    start_test(args)
    #Script produces 2 output files

    # Put them all together (revise)
    phs = "session" if args.dataset.lower() == "ldp" else "phase"
    command = "cd {};".format(args.outputpath)
    print(command)
    # os.system(command)
    command = "head -1 BERT-" + args.dataset.upper() + "-incr" + phs + "-3_" + args.who + "_testsession3_nouns_w_2_*.csv" + \
              "> all_nw2dets_incr_bert_" + args.who + "{}.csv;".format("_downsampled" if args.downsampled_parents else "")
    print(command)
    # os.system(command)
    command = "for f in BERT-" + args.dataset.upper() + "-incrsession-*_" + args.who + "_test*_nouns_w_2_*.csv; do sed 1d $f >> all_nw2dets_incr_bert_" + args.who + ".csv; done".format(
        phs)
    print(command)
    # os.system(command)
