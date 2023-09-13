# coding:utf-8
"""
Name : ldpdata.py
Author : Raquel G. Alhama
Desc:
"""
import string
import os, sys, inspect
import pandas as pd
import random
SCRIPT_FOLDER = os.path.realpath(os.path.abspath(
    os.path.split(inspect.getfile(inspect.currentframe()))[0]))
PROJECT_FOLDER = os.path.join(SCRIPT_FOLDER, os.pardir, os.pardir)
sys.path.insert(0, PROJECT_FOLDER)
DATA_PATH=os.path.join(PROJECT_FOLDER, "data")
from src.datahandlers.vocab import build_vocab_from_tokenized
# Create a blank Tokenizer with just the English vocab
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English()
en_tokenizer = Tokenizer(nlp.vocab)

class LDPData:
    def __init__(self):
        self.utts = self.load_utts_df()
        self.tokenized=self.get_tokenized_utts()
        self.vocab=self.get_vocab()
        self.subjects=self.utts['subject'].unique()

    #Methods for all utterances (not per session)
    #Some are abstract and need to be implemented for child or parent
    def load_utts_df(self):
        pass

    def get_all_utts(self):
        pass

    def get_tokenized_utts(self):
        #Tokenize and remove punctuation
        return [en_tokenizer(u.strip(string.punctuation).strip()) for u in self.get_all_utts()]

    def get_ntokens(self):
        return sum([len(s) for s in self.get_all_utts()])

    def get_utts_dets(self):
        pass

    def get_tokenized_utts_dets(self):
        return [en_tokenizer(u.strip(string.punctuation).strip()) for u in self.get_utts_dets()]

    #Methods for utterances/session
    #Some are abstract and need to be implemented for child or parenet
    def get_utts_session(self, nsession):
        pass

    def get_tokenized_utts_session(self, nsession):
        return [en_tokenizer(u.strip(string.punctuation).strip()) for u in self.get_utts_session(nsession)]


    def get_vocab(self):
        return build_vocab_from_tokenized(self.tokenized)

    #Methods to measure productivity during the sessions
    def load_session_counts(self):
        df = pd.read_csv(os.path.join(PROJECT_FOLDER, "data/processed/LDP/child_ranges_by_session.csv"), sep=";")
        return df

    def get_median_utts_session(self, session):
        self.session_prod = self.load_session_counts()
        return int(self.session_prod[self.session_prod.session == session].median_utterances)

    def _get_rows_to_mask(self, session, column):
        df_det = self.utts[self.utts[column].notnull()]
        if session is not None:
            df_det = df_det[df_det['session'] == session]
        zdet = zip(df_det.session, df_det.subject, df_det[self.own_utts_column], df_det[column])
        return zdet

    def get_masked_dets(self, column, session, filter_dets, mask, samplesize=-1):
        masked=[]
        zdet=list(self._get_rows_to_mask(session, column))
        if 1 < samplesize < len(zdet): #downsample
            zdet=random.sample(zdet,samplesize)
        for session, subject, s, ph in zdet:
            sl = s.strip().split()
            phl = ph.strip().split()
            det, noun=str(phl[0]), str(phl[1])
            if len(filter_dets)== 0 or (det in filter_dets):
                for i,w in enumerate(sl):
                    if w.lower() == det.lower():
                        if ((i+1) < len(sl) and (sl[i+1] == noun)):
                            sl[i]=mask
                            masked.append((session, subject, ' '.join(sl), w.lower(), noun.lower()))
                            break
                        elif ((i + 1) < len(sl) and (sl[i + 1].lower() == w.lower())): #det repetition
                            break
                        elif ((i + 2) < len(sl) and (sl[i + 2].lower() == noun.lower())):
                            sl[i]=mask
                            masked.append((session, subject, ' '.join(sl), w.lower(), noun.lower()))
                            break
        return masked

class LDPParents(LDPData):
    def __init__(self):
        super().__init__()
        self.own_utts_column="p_utts"

    def load_utts_df(self):
        df = pd.read_csv(os.path.join(PROJECT_FOLDER, "data/processed/LDP/parents_utterances_session.csv"), sep=";")
        return df

    def get_all_utts(self):
        return list(self.utts.p_utts)

    def get_utts_dets(self):
        utts_w_det = self.utts[self.utts.parent_det_strs_indef_def_LEMMA.notnull()]
        return list(utts_w_det.p_utts)

    def get_total_det_utts_subject_session(self):
        utts_w_det = self.utts[self.utts.parent_det_strs_indef_def_LEMMA.notnull()]
        g=utts_w_det.groupby(["session","subject"])["c_utts"].count().reset_index()
        return g

    def get_utts_session(self, nsession):
        return self.utts[self.utts.session == nsession]["p_utts"]

    def get_masked_dets(self, session=None, mask='[MASK]', column='parent_det_strs_indef_def_LEMMA', filter_dets=[], samplesize=-1):
        return super().get_masked_dets(column, session, filter_dets, mask, samplesize=samplesize)

class LDPChildren(LDPData):
    def __init__(self):
        super().__init__()
        self.own_utts_column="c_utts"

    def load_utts_df(self):
        df = pd.read_csv(os.path.join(PROJECT_FOLDER, "data/processed/LDP/children_utterances_session.csv"), sep=";")
        return df

    def get_all_utts(self):
        return list(self.utts.c_utts)

    def get_utts_dets(self):
        utts_w_det = self.utts[self.utts.child_det_strs_indef_def_LEMMA.notnull()]
        return list(utts_w_det.c_utts)

    def get_number_utts_dets_session(self, session):
        utts_w_det = self.utts[self.utts.child_det_strs_indef_def_LEMMA.notnull()]
        utts_w_det = utts_w_det[self.utts.session == session]
        return len(utts_w_det)
    
    def get_total_det_utts_subject_session(self):
        utts_w_det = self.utts[self.utts.child_det_strs_indef_def_LEMMA.notnull()]
        g=utts_w_det.groupby(["session","subject"])["c_utts"].count().reset_index()
        return g

    def get_utts_session(self, nsession):
        return self.utts[self.utts.session == nsession][self.own_utts_column]

    def get_masked_dets(self, session=None, mask='[MASK]', column='child_det_strs_indef_def_LEMMA', filter_dets=[], samplesize=-1):
        return super().get_masked_dets(column, session, filter_dets, mask, samplesize)


class LDPChildrenOmissions(LDPData):

    def __init__(self):
        self.utts=self.load_utts_df()
        self.own_utts_column="c_utts"

    def load_utts_df(self):
        df = pd.read_csv(os.path.join(PROJECT_FOLDER, "data/processed/LDP/child_null_det_slots.csv"), sep=",")
        return df

#Test
if __name__ == '__main__':
    c=LDPChildrenOmissions()
    print(c.utts)