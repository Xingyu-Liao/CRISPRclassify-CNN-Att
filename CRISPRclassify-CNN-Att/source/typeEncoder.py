import pandas as pd
class TypeEncoder(object):
    def __init__(self,type_selected_list) :
        self.type_list = type_selected_list
    # 对类别进行简单的编码
    def encode_type_3(self,smallorbig):
        type_list = self.type_list
        if smallorbig == 0:
            type_final = ['VI-A','V-K','II-B','V-F1','V-F2','VI-D','V-B1','VI-B2','VI-B1','IV-A3','I-U']
        if smallorbig ==1:
            type_final = ['I-E','I-C','II-A','I-F','I-G','V-A','II-C','I-D','I-B','III-A','I-A']   
        if smallorbig == 2:
            type_final = ['I-E','I-C','II-A','I-F','I-G','V-A','II-C','I-D','I-B','III-A','I-A']+['VI-A','V-K','II-B','V-F1','V-F2','VI-D','V-B1','VI-B2','VI-B1','IV-A3','I-U']
        type_encoded_all=[]
        for i in range(len(type_list)):
            type_encoded_all.append(type_final.index(type_list[i]))
        return type_encoded_all


