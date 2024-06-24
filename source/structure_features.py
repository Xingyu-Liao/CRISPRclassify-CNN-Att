import pandas as pd
import numpy as np
import itertools as it
from dataselect import DataSelect
from sklearn.preprocessing import MinMaxScaler, StandardScaler
class RepeatFeature(object):
    def __init__(self):
        self.input_file = "repeats.fa"
        self.data = pd.DataFrame(columns=['Seq', 'Type'])
        self.kmer = 6
        self.comp_tab = str.maketrans("ACGT", "TGCA")
    def convert_fasta(self):
        sequences = []
        categories = []
        with open(self.input_file, 'r') as file:
            lines = file.readlines()
            # print(lines)
            current_sequence = ""
            current_category = ""
            for line in lines:
                if line.startswith('>'):
                    if current_sequence:
                        sequences.append(current_sequence)
                        categories.append(current_category)
                        current_sequence = ""

                    current_category = line.strip()[1:]
                else:
                    current_sequence += line.strip()
            if current_sequence:
                sequences.append(current_sequence)
                categories.append(current_category)
        df = pd.DataFrame({'Seq': sequences, 'Type': categories})
        self.data = df
        
    def read_data(self,type_size):   
        dataselect = DataSelect()
        sequences_list,type_list = dataselect.count_class_num(type_size)
        df = pd.DataFrame({'Seq': sequences_list, 'Type': type_list})
        self.data = df


    def count_kmer(self, seq):
        kmer_d = {}
        for i in range(len(seq) - self.kmer + 1):
            kmer_for = seq[i:(i + self.kmer)]
            kmer_rev = kmer_for.translate(self.comp_tab)[::-1]
            if kmer_for < kmer_rev:
                kmer = kmer_for
            else:
                kmer = kmer_rev
            if kmer in kmer_d:
                kmer_d[kmer] += 1
            else:
                kmer_d[kmer] = 1
        return kmer_d

    def generate_canonical_kmer(self):
        letters = ['A', 'C', 'G', 'T']
        all_kmer = [''.join(k) for k in it.product(letters, repeat=self.kmer)]
        all_kmer_rev = [x.translate(self.comp_tab)[::-1] for x in all_kmer]
        can_kmer = list(it.compress(all_kmer_rev, [not kf < kr for kf, kr in zip(all_kmer, all_kmer_rev)]))
        can_kmer.sort()
        self.can_kmer = can_kmer

    def prepare_data(self):
        self.generate_canonical_kmer()
        X = pd.DataFrame([dict(zip(self.can_kmer, np.zeros(len(self.can_kmer))))] + [self.count_kmer(x) for x in
                                                                                     self.data['Seq']]).fillna(0)
        X = X.iloc[1:]
        X['Length'] = [len(x) for x in self.data['Seq']]
        X['GC'] = [(x.count('G') + x.count('C')) / len(x) for x in self.data['Seq']]
        X = X.iloc[:, :2082]   
        return  X
