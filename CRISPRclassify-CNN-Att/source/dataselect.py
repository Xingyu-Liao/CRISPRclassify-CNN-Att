import pandas as pd
from openpyxl import Workbook
from collections import Counter
import os
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
csv_path = os.path.join(project_root, 'data', 'repeats_all.csv')

class DataSelect(object):
    def __init__(self):
        
        self.type_list_all,self.sequences_list_all = self.read_repeats()
        self.type_have_cas=['I-A','I-B','I-C','I-D','I-E','I-F1','I-F2','I-F3','I-G','I-F','I-U',
                            'II-A','II-B','II-C',
                            'III-A','III-B','III-C','III-D','III-E','III-F',
                            'IV-A1','IV-A2','IV-A3','IV-B','IV-C','IV-D','IV-E',
                            'V-A','V-B1','V-B2','V-C','V-D','V-E','V-F1','V-F2','V-F3','V-G','V-H','V-I','V-K','V-F','V-J',
                            'VI-A','VI-B1','VI-B2','VI-C','VI-D'
                            ]

    def read_repeats(self):
        df = pd.read_csv(csv_path)
        type_list = df['type'].tolist()
        sequences_list = df['seq'].tolist()
        type_counts = Counter(type_list)

        return type_list,sequences_list

    def count_class_num(self, type_size):
        types_all = self.type_list_all
        sequences_all = self.sequences_list_all
        type_counts = Counter(types_all)
        selected_index = []
        for i in range(len(types_all)):
            type_selected_big = ['I-A', 'I-B', 'I-C', 'I-D', 'I-E', 'I-F', 'I-G', 'I-U', 'II-A', 'II-C', 'V-A']
            type_selected_small = ['II-B', 'III-C', 'IV-A1', 'IV-A2', 'IV-A3', 'IV-D', 'IV-E', 'V-B1', 'V-B2', 'V-F1', 'V-F2', 'V-F3', 'V-K', 'VI-A','VI-B1', 'VI-B2', 'VI-C', 'VI-D']

            type_selected_all = ['I-A','I-B', 'I-C', 'I-D','I-E', 'I-F','I-G','I-U',
            'II-A','II-B','II-C',
             'III-C',
             'IV-A1','IV-A2','IV-A3','IV-D','IV-E',
             'V-A','V-B1','V-B2','V-F1','V-F2','V-F3','V-K',
             'VI-A', 'VI-B1','VI-B2','VI-C','VI-D'
            ]

            if type_size == 0:
                type_selected_final = type_selected_small
            elif type_size ==1:
                type_selected_final = type_selected_big
            elif type_size ==2:
                type_selected_final = type_selected_all

            if types_all[i] in self.type_have_cas and types_all[i] in type_selected_final:
                selected_index.append(i)

        type_selected=[]
        seq_selected=[]
        for i in range(len(selected_index)):
            type_selected.append(types_all[selected_index[i]])
            seq_selected.append(sequences_all[selected_index[i]])
        repeat_type_dict = {}
        for i, seq_selected in enumerate(seq_selected):
            if seq_selected not in repeat_type_dict:
                repeat_type_dict[seq_selected] = type_selected[i]
        unique_repeats = list(repeat_type_dict.keys())
        unique_types = list(repeat_type_dict.values())
        return unique_repeats,unique_types