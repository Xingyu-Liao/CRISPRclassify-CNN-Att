import pandas as pd
import numpy as np
class RepeatEncoder(object):
    def __init__(self,repeats_selected_list):
        self.repeats_list = repeats_selected_list
        self.repeats_filled = self.fill_repeats()
    def fill_repeats(self):
        sequences = self.repeats_list
        max_len = 48
        sequences_filled = [seq.ljust(max_len, 'X') for seq in sequences]
        return sequences_filled
    # ATCG编码为四通道的
    def repeats_onehot_encoder(self):
        all_encodings=[]
        for seq in self.repeats_filled:
            e=[]
            for i in seq:
                if i == 'X':
                    l=[0,0,0,0]
                elif i == 'A':
                    l=[1,0,0,0]
                elif i == 'T':
                    l=[0,1,0,0]
                elif i =='C':
                    l=[0,0,1,0]
                elif i =='G':
                    l=[0,0,0,1]
                else:
                    l=[0,0,0,0]
                e.append(l)
            array=np.array(e)
            array_T = array.T
            all_encodings.append(array_T)
        return all_encodings