import pandas as pd
class TypeEncoder(object):
    def __init__(self,type_selected_list) :
        self.type_list = type_selected_list
    # 对类别进行简单的编码
    def encode_type_3(self,smallorbig):
        type_list = self.type_list
        if smallorbig == 0:
            
            type_final = ['II-B', 'III-C', 'IV-A1', 'IV-A2', 'IV-A3', 'IV-D', 'IV-E', 'V-B1', 'V-B2', 'V-F1', 'V-F2', 'V-F3', 'V-K', 'VI-A','VI-B1', 'VI-B2', 'VI-C', 'VI-D']
        if smallorbig ==1:
            type_final = ['I-A', 'I-B', 'I-C', 'I-D', 'I-E', 'I-F', 'I-G', 'I-U', 'II-A', 'II-C', 'V-A']
        if smallorbig == 2:
            # type_final =  ['I-E', 'I-C', 'I-B', 'III-A', 'II-A', 'I-F', 'III-B', 'III-D', 'I-G', 'V-A', 'II-C', 'I-D', 'I-A', 
            # 'I-U', 'VI-A', 'V-F', 'II-B', 'V-F1', 'V-K', 'III-C', 'IV-A1', 'VI-B1', 'IV-A3', 'V-F2', 'VI-D', 'V-B1', 'IV-A2', 'VI-B2', 'V-J']

            type_final = ['I-A','I-B', 'I-C', 'I-D','I-E', 'I-F','I-G','I-U',
            'II-A','II-B','II-C',
             'III-C',
             'IV-A1','IV-A2','IV-A3','IV-D','IV-E',
             'V-A','V-B1','V-B2','V-F1','V-F2','V-F3','V-K',
             'VI-A', 'VI-B1','VI-B2','VI-C','VI-D'
            ]
        type_encoded_all=[]
        for i in range(len(type_list)):
            type_encoded_all.append(type_final.index(type_list[i]))
        return type_encoded_all


