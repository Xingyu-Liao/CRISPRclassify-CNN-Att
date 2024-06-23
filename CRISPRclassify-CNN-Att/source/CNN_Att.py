import torch
import torch.nn as nn
import torch.optim as optim
from repeatsEncoder import RepeatEncoder
from typeEncoder import TypeEncoder
from dataselect import DataSelect
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import xgboost as xgb
from tqdm import tqdm
from structure_features import RepeatFeature
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.preprocessing import label_binarize
from collections import Counter
import math
from scipy.spatial.distance import euclidean
from itertools import product

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
        X_tensor = features_tensor.unsqueeze(0)
        bio_feature_tensor = bio_feature_tensor.unsqueeze(0)
        return X_tensor, bio_feature_tensor,torch.tensor(label, dtype=torch.long)

    def char_to_index(self, char):
        char_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'X': 4,'K':0,'Y':0,'R':0,'M':0}
        if char in char_dict:
            return char_dict[char]

class CNNTrainer:
    def __init__(self, model,optimizer, class_names, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.class_names = class_names
        self.best_accuracy = 0.0
        self.best_model_path = 'cnn_att_large.pth'  
    def train(self, train_dataloader, test_dataloader,num_epochs=50 ):
        criterion = nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            epoch_train_loss = 0.0
            epoch_test_loss = 0.0
            train_correct = 0
            test_correct = 0
            total_train = 0
            total_test = 0
            class_correct = list(0.0 for _ in range(len(self.class_names)))
            class_total = list(0.0 for _ in range(len(self.class_names)))
            all_labels = []
            all_predicted = []
            # 训练模型
            self.model.train()
            for inputs, bio_features, labels in train_dataloader:
                inputs, bio_features, labels = inputs.to(self.device), bio_features.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs, bio_features)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)


                train_correct += (predicted == labels).sum().item()
                total_train += labels.size(0)
            epoch_train_loss /= len(train_dataloader)
            train_accuracy = train_correct / total_train 
            # 测试模型
            self.model.eval()
            with torch.no_grad():
                for inputs, bio_features, labels in test_dataloader:
                    inputs, bio_features, labels = inputs.to(self.device), bio_features.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs, bio_features)
                    loss = criterion(outputs, labels)
                    epoch_test_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    test_correct += (predicted == labels).sum().item()
                    total_test += labels.size(0)
                    c = (predicted == labels).squeeze()  
                    for i in range(labels.size(0)):
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1
                        all_labels.append(label.item())
                        all_predicted.append(predicted[i].item())

            epoch_test_loss /= len(test_dataloader)
            test_accuracy = test_correct / total_test

            if test_accuracy > self.best_accuracy:
                self.best_accuracy = test_accuracy
                torch.save(self.model.state_dict(), self.best_model_path)
                print("Best model saved at:", self.best_model_path)
            print("--------------------------------")
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f},Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
            # 打印每个类别的acc，precision，recall，F1指标
            for i in range(len(self.class_names)):
                if class_total[i] > 0:
                    acc = class_correct[i] / class_total[i]
                    precision = precision_score(all_labels, all_predicted, labels=[i], average=None)[0]
                    recall = recall_score(all_labels, all_predicted, labels=[i], average=None)[0]
                    f1 = f1_score(all_labels, all_predicted, labels=[i], average=None)[0]
                    print(f'Class {self.class_names[i]} - Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
                else:
                    print(f'Class {self.class_names[i]} - Acc: N/A (no samples), Precision: N/A, Recall: N/A, F1: N/A')
            accuracy = accuracy_score(all_labels, all_predicted)
            precision = precision_score(all_labels, all_predicted, average='weighted')
            recall = recall_score(all_labels, all_predicted, average='weighted')
            f1 = f1_score(all_labels, all_predicted, average='weighted')
            all_labels_np = np.array(all_labels)
            all_predicted_np = np.array(all_predicted)
            print(f'Overall Test Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        attn_weights = self.softmax(self.linear2(torch.tanh(self.linear1(x))))
        weighted_input = torch.mul(x, attn_weights)
        return weighted_input
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, seq_length, bio_feature_dim):
        super(CNNClassifier, self).__init__()
        attn_hidden_dim = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=(2,4))
        self.conv2 = nn.Conv1d(64, 64, kernel_size=(2,4))
        self.conv3 = nn.Conv1d(64, 64, kernel_size=(2,4)) 
        self.conv_bio1 = nn.Conv1d(1, 64, kernel_size=(1,5))
        self.conv_bio2 = nn.Conv1d(64, 64, kernel_size=(1,11))
        self.conv_bio3 = nn.Conv1d(64, 64, kernel_size=(1,17))
        self.seq_length = seq_length
        self.bio_feature_dim = bio_feature_dim
        self.fc_seq = nn.Linear(64 * (seq_length - 9), 512)
        self.fc_bio = nn.Linear(64 * (bio_feature_dim - 30), 512)  
        self.attn_seq = SelfAttention(512, attn_hidden_dim) 
        self.attn_bio = SelfAttention(512, attn_hidden_dim)  
        self.attn_combined = SelfAttention(1024, attn_hidden_dim) 
        self.fc_final = nn.Linear(1024, num_classes)  
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, bio_features):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv3(x))  
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1) 
        bio_features = bio_features.unsqueeze(1)  
        bio_features = nn.functional.relu(self.conv_bio1(bio_features))
        bio_features = nn.functional.relu(self.conv_bio2(bio_features))
        bio_features = nn.functional.relu(self.conv_bio3(bio_features))
        bio_features = bio_features.view(bio_features.size(0), -1) 
        x_seq = nn.functional.relu(self.fc_seq(x))
        x_bio = nn.functional.relu(self.fc_bio(bio_features))
        x_seq_attended = self.attn_seq(x_seq)
        x_bio_attended = self.attn_bio(x_bio)
        x_combined = torch.cat((x_seq_attended, x_bio_attended), dim=1)
        x_combined_attended = self.attn_combined(x_combined)
        x = self.dropout(x_combined_attended)
        feature = x
        x = self.fc_final(x)    
        probabilities = torch.softmax(x, dim=1)
        return x
if __name__ == '__main__':
    dataselect = DataSelect()
    seq_selected,type_selected = dataselect.count_class_num(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    repeatsencoder = RepeatEncoder(seq_selected)
    seq_filled = repeatsencoder.fill_repeats()
    X2 = seq_filled

    # 用四聚体作为特征
    repeatfeature = RepeatFeature()
    repeatfeature.read_data(1)
    X1 = repeatfeature.prepare_data()
    repeatencoder = RepeatEncoder(seq_selected)
    # 2: 四通道编码
    X2 = repeatencoder.repeats_onehot_encoder()
    X1.reset_index(drop=True, inplace=True)
    bio_features = X1
    bio_feature_dim = 2082
    typeencoder = TypeEncoder(type_selected)
    Y = typeencoder.encode_type_3(1)
    Y_df = pd.DataFrame(Y, columns=['Y'])
    type_selected_big = ['I-E','I-C','II-A','I-F','I-G','V-A','II-C','I-D','I-B','III-A']
    class_names = type_selected_big

    embedding_dim = 64
    vocab_size = 5
    num_classes = 10
    seq_length = 48

    x_train, x_temp, y_train, y_temp,bio_features_train,bio_features_test = train_test_split(X2, Y, bio_features,test_size=0.5,random_state=15)
    train_dataset = MyDataset(x_train,bio_features_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = MyDataset(x_temp, bio_features_test,y_temp)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    model = CNNClassifier(vocab_size, embedding_dim, num_classes, seq_length, bio_feature_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)
    trainer = CNNTrainer(model, optimizer,class_names, device)
    trainer.train(train_dataloader, test_dataloader)
 