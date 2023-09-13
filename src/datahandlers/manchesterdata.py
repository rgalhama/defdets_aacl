# coding:utf-8
"""
Name : manchesterdata.py
Author : Raquel G. Alhama
Desc:
"""
import string
import os, sys, inspect
import pandas as pd
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


class ManchesterData:
    def __init__(self):
        self.df_phases = pd.read_csv(os.path.join(PROJECT_FOLDER,
                                                  "data/processed/Manchester/phases.csv"), sep=",")

        pd.to_numeric(self.df_phases.upper_limit)
        pd.to_numeric(self.df_phases.Phase)
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
    #Some are abstract and need to be implemented for child or parent (may be overriden)
    def get_utts_session(self, nsession):
        pass

    def get_tokenized_utts_session(self, nsession):
        return [en_tokenizer(u.strip(string.punctuation).strip()) for u in self.get_utts_session(nsession)]


    def get_vocab(self):
        return build_vocab_from_tokenized(self.tokenized)

    #Methods to measure productivity during the sessions/phases
    def load_session_counts(self):
        df = pd.read_csv(os.path.join(PROJECT_FOLDER, "data/processed/LDP/child_ranges_by_session.csv"), sep=";")
        return df

    def get_median_utts_session(self, session):
        self.session_prod = self.load_session_counts()
        return int(self.session_prod[self.session_prod.session == session].median_utterances)

    def get_masked_dets(self, column, session, filter_dets, mask):
        df_det = self.utts[self.utts[column].notnull()]
        if session is not None:
            df_det=df_det[df_det['phase'] == session]
        zdet = zip(df_det.session, df_det.subject, df_det[self.own_utts_column], df_det[column])
        masked=[]
        for session, subject, s, ph in zdet:
            sl = s.strip().split()
            phl = ph.strip().split()
            det, noun=str(phl[0]), str(phl[1])
            if len(filter_dets)== 0 or (det in filter_dets):
                for i,w in enumerate(sl):
                    if w == det:
                        if ((i+1) < len(sl) and (sl[i+1] == noun)):
                            sl[i]=mask
                            masked.append((session, subject, ' '.join(sl), w, noun))
                            break
                        elif ((i + 1) < len(sl) and (sl[i + 1] == w)): #det repetition
                            break
                        elif ((i + 2) < len(sl) and (sl[i + 2] == noun)):
                            sl[i]=mask
                            masked.append((session, subject, ' '.join(sl), w, noun))
                            break
        return masked


class ManchesterChildren(ManchesterData):
    def __init__(self):
        self.own_utts_column="child_utterances"
        super().__init__()

    def load_utts_df(self):
        chfname="data/processed/Manchester/manchester_child_lemma_nps_data_NO_PLURALS.csv"
        df = pd.read_csv(os.path.join(PROJECT_FOLDER, chfname), sep=";")
        df=df.fillna('')
        #Create phase column if it doesn't exist
        if 'phase' not in df.columns:
            df['phase']=[0]*len(df)
            df['phase'] = df.apply(lambda x: self.phase_from_age(x.subject,x['manchester_age']), axis=1)
            df.to_csv(os.path.join(PROJECT_FOLDER, chfname), sep=";")
        #Join def/indef columns (separated in this dataset)
        if 'child_det_strs_indef_def_LEMMA' not in df.columns:
            df['child_det_strs_indef_def_LEMMA'] = df.apply(lambda x: '%s%s' % (x['def_np_lemma'], x['indef_np_lemma']), axis=1)
            aux = df.apply(lambda x : True
                if len(x['child_det_strs_indef_def_LEMMA'].split(' '))>2 else False, axis=1)
            num_rows = len(aux[aux == True].index)
            if num_rows > 1:
                raise Exception("Error processing datafile. Some utterances have more than one determinant.")
            df.to_csv(os.path.join(PROJECT_FOLDER, chfname), sep=";")
        return df

    def phase_from_age(self, subject, age):
        months=int(age[:2])*12+int(age[2:4])
        #get phase for month
        phases_child = self.df_phases[self.df_phases.Child == subject]
        phases=phases_child[months<=phases_child.upper_limit]['Phase']
        phases=pd.to_numeric(phases)
        phase=phases.min()
        return phase

    def get_all_utts(self):
        return list(self.utts[self.own_utts_column])

    def get_utts_dets(self):
        utts_w_det = self.utts[self.utts.child_det_strs_indef_def_LEMMA.notnull()]
        return list(utts_w_det[self.own_utts_column])

    def get_utts_session(self, nsession):
        return self.utts[self.utts.session == nsession][self.own_utts_column]

    def get_masked_dets(self, session=None, mask='[MASK]', column='child_det_strs_indef_def_LEMMA', filter_dets=[]):
        self.utts[column].apply(str)
        df_det = self.utts[self.utts[column]!='']
        if session is not None:
            df_det=df_det[df_det['phase'] == session]
        zdet = zip(df_det.phase, df_det.subject, df_det[self.own_utts_column], df_det[column])
        masked=[]
        for session, subject, s, ph in zdet:
            sl = s.strip().split()
            phl = ph.strip().split()
            det, noun=str(phl[0]), str(phl[1])
            if len(filter_dets)== 0 or (det in filter_dets):
                for i,w in enumerate(sl):
                    if w == det:
                        if ((i+1) < len(sl) and (sl[i+1] == noun)):
                            sl[i]=mask
                            masked.append((session, subject, ' '.join(sl), w, noun))
                            break
                        elif ((i + 1) < len(sl) and (sl[i + 1] == w)): #det repetition
                            break
                        elif ((i + 2) < len(sl) and (sl[i + 2] == noun)):
                            sl[i]=mask
                            masked.append((session, subject, ' '.join(sl), w, noun))
                            break
        return masked


#Test
if __name__ == '__main__':
    p=ManchesterChildren()
    m=p.get_masked_dets(3, column='child_det_strs_indef_def_LEMMA')
    for t in m:
        print(t)
