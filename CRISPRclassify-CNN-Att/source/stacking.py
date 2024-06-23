import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import xgboost as xgb
from torch.utils.data import ConcatDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from repeatsEncoder import RepeatEncoder
from typeEncoder import TypeEncoder
from dataselect import DataSelect
from structure_features import RepeatFeature
from CNN_Att import CNNClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import pandas as pd
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, bio_features,y):
        self.X = X
        self.y = y
        self.bio_features = bio_features

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        features = self.X[index]
        label = self.y[index]
        bio_feature =  self.bio_features.iloc[index]
        features_tensor = torch.tensor(features, dtype=torch.float)
        bio_feature_tensor = torch.tensor(bio_feature, dtype=torch.float)
        # 添加一个维度表示序列长度
        X_tensor = features_tensor.unsqueeze(0)
        bio_feature_tensor = bio_feature_tensor.unsqueeze(0)
        return X_tensor, bio_feature_tensor,torch.tensor(label, dtype=torch.long)

if __name__ == '__main__':
    dataselect = DataSelect()
    seq_selected,type_selected_small = dataselect.count_class_num(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    repeatfeature = RepeatFeature()
    repeatfeature.read_data(0)
    X1 = repeatfeature.prepare_data()
    X1.reset_index(drop=True, inplace=True)
    bio_features_small = X1
    repeatencoder = RepeatEncoder(seq_selected)
    X2_small = repeatencoder.repeats_onehot_encoder()
    typeencoder = TypeEncoder(type_selected_small)
    type_small = ['VI-A','V-K','II-B','V-F1','V-F2','VI-D','V-B1','VI-B2','VI-B1','I-A','IV-A3','I-U']
    Y_small = typeencoder.encode_type_3(0)
    for i in range(len(Y_small)):
        Y_small[i] = Y_small[i] + 10
    x_train_small, x_temp_small, y_train_small, y_temp_small,bio_features_train_small,bio_features_test_small = train_test_split(X2_small, Y_small, bio_features_small,test_size=0.5,random_state=15)
    test_dataset_small = MyDataset(x_temp_small, bio_features_test_small,y_temp_small)
    dataselect = DataSelect()
    seq_selected,type_selected_big = dataselect.count_class_num(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    repeatfeature = RepeatFeature()
    repeatfeature.read_data(1)
    X1 = repeatfeature.prepare_data()
    X1.reset_index(drop=True, inplace=True)
    bio_features_big = X1
    repeatencoder = RepeatEncoder(seq_selected)
    X2_big = repeatencoder.repeats_onehot_encoder()
    typeencoder = TypeEncoder(type_selected_big)
    Y_big = typeencoder.encode_type_3(1)
    type_big = ['I-E','I-C','II-A','I-F','I-G','V-A','II-C','I-D','I-B','III-A']
    x_train_big, x_temp_big, y_train_big, y_temp_big,bio_features_train_big,bio_features_test_big = train_test_split(X2_big, Y_big, bio_features_big,test_size=0.5,random_state=15)
    test_dataset_big = MyDataset(x_temp_big, bio_features_test_big, y_temp_big)
    concat_dataset = ConcatDataset([test_dataset_big, test_dataset_small])
    test_dataloader = DataLoader(concat_dataset, batch_size=32, shuffle=False)
    cnn_model_state_dict = torch.load('cnn_att_large.pth')
    cnn_2_model_state_dict = torch.load('cnn_att_less.pth')

    embedding_dim = 64
    vocab_size = 5
    num_classes_small = 12
    num_classes_big = 10
    seq_length = 48
    bio_feature_dim = 2082

    cnn_model = CNNClassifier(vocab_size, embedding_dim, num_classes_big, seq_length, bio_feature_dim).to(device)  
    cnn_2_model = CNNClassifier(vocab_size, embedding_dim, num_classes_small, seq_length, bio_feature_dim).to(device) 

    cnn_model.load_state_dict(cnn_model_state_dict)
    cnn_2_model.load_state_dict(cnn_2_model_state_dict)


    cnn_model.eval()
    cnn_2_model.eval()


    predictions_cnn = []
    predictions_cnn_2 = []

    with torch.no_grad():
        for inputs, bio_features, labels in test_dataloader:
            inputs, bio_features, labels = inputs.to(device), bio_features.to(device), labels.to(device)
            outputs_cnn = cnn_model(inputs, bio_features)
            predictions_cnn.extend(outputs_cnn.cpu().numpy())
            outputs_cnn_2 = cnn_2_model(inputs, bio_features)
            predictions_cnn_2.extend(outputs_cnn_2.cpu().numpy())
    new_features = np.column_stack((predictions_cnn, predictions_cnn_2))
    x_temp_combined = np.concatenate((x_temp_big, x_temp_small), axis=0)
    y_temp_combined = np.concatenate((y_temp_big, y_temp_small), axis=0)
    
    bio_features_combined = np.concatenate((bio_features_test_big, bio_features_test_small),axis=0)

    X_train_stacked, X_test_stacked, y_train_stacked, y_test_stacked = train_test_split(
        np.concatenate((bio_features_combined, new_features), axis=1), y_temp_combined, test_size=0.5)

    xgb_meta_model = xgb.XGBClassifier(objective='multi:softprob', num_class=22, max_depth=16)
    xgb_meta_model.fit(X_train_stacked, y_train_stacked)

    final_predictions = xgb_meta_model.predict(X_test_stacked)
    proba_predictions = xgb_meta_model.predict_proba(X_test_stacked)
    accuracy = accuracy_score(y_test_stacked, final_predictions)
    acc_test=0
    for i in range(len(y_test_stacked)):
        if y_test_stacked[i] == final_predictions[i]:
            acc_test = acc_test+1
    precision = precision_score(y_test_stacked, final_predictions, average='weighted')
    recall = recall_score(y_test_stacked, final_predictions, average='weighted')
    f1 = f1_score(y_test_stacked, final_predictions, average='weighted')
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    report = classification_report(y_test_stacked, final_predictions)
    print(report)

    auc_per_class = []
    for class_idx in range(22):
        y_true_class = [1 if label == class_idx else 0 for label in y_test_stacked]
        proba_predictions_class = proba_predictions[:, class_idx]
        auc_class = roc_auc_score(y_true_class, proba_predictions_class)
        auc_per_class.append(auc_class)
    acc_per_class = []
    for class_idx in range(22):
        final_predictions_class = [1 if label == class_idx else 0 for label in y_test_stacked]
        y_true_class = [1 if label == class_idx else 0 for label in final_predictions]
        acc_class = accuracy_score(y_true_class, final_predictions_class)
        acc_per_class.append(acc_class)