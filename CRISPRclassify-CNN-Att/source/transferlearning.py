import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from CNN_Att import CNNClassifier,SelfAttention 
from repeatsEncoder import RepeatEncoder
from typeEncoder import TypeEncoder
from dataselect import DataSelect
from structure_features import RepeatFeature
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import copy
from sklearn.metrics import classification_report
from torch.utils.data import ConcatDataset
import os
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
model_dir = os.path.join(project_root, 'model')
model_path_large = os.path.join(model_dir, 'cnn_att_large.pth')
model_path_less= os.path.join(model_dir, 'cnn_att_less.pth')
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fine_tune_model(model, train_loader, test_loader, num_classes, num_epochs=100, learning_rate=0.001):
    model.to(device)
    best_accuracy = 0.0
    best_model_params = None
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, bio_features, labels in train_loader:
            inputs, bio_features, labels = inputs.to(device), bio_features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, bio_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        accuracy = evaluate_model(model, test_loader)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_params = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_params)
    torch.save(model.state_dict(), model_path_less)

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, bio_features, labels in test_loader:
            inputs, bio_features, labels = inputs.to(device), bio_features.to(device), labels.to(device)
            outputs = model(inputs, bio_features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    accuracy = (correct / total) * 100
    print(f"Accuracy on test set: {accuracy}%")
    report = classification_report(all_labels, all_predictions, target_names=[str(i) for i in range(num_classes)])
    print(report)
    return accuracy

dataselect = DataSelect()
seq_selected,type_selected = dataselect.count_class_num(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

repeatsencoder = RepeatEncoder(seq_selected)
seq_filled = repeatsencoder.fill_repeats()
X2 = seq_filled

# 用四聚体作为特征
repeatfeature = RepeatFeature()
repeatfeature.read_data(0)
X1 = repeatfeature.prepare_data()
repeatencoder = RepeatEncoder(seq_selected)
# 2: 四通道编码
X2 = repeatencoder.repeats_onehot_encoder()
X1.reset_index(drop=True, inplace=True)
bio_features = X1
bio_feature_dim = 2082
typeencoder = TypeEncoder(type_selected)
Y = typeencoder.encode_type_3(0)
Y_df = pd.DataFrame(Y, columns=['Y'])
type_selected_big = ['I-E','I-C','II-A','I-F','I-G','V-A','II-C','I-D','I-B','III-A','I-A']
type_selected_small = ['VI-A','V-K','II-B','V-F1','V-F2','VI-D','V-B1','VI-B2','VI-B1','IV-A3','I-U']
embedding_dim = 64
vocab_size = 5
num_classes = 11
bio_feature_dim=2082
seq_length = 48
class_names = type_selected_small
x_train, x_temp, y_train, y_temp,bio_features_train,bio_features_test = train_test_split(X2, Y, bio_features,test_size=0.5,random_state=15)
train_dataset = MyDataset(x_train,bio_features_train, y_train)
test_dataset = MyDataset(x_temp, bio_features_test,y_temp)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
model = CNNClassifier(vocab_size, embedding_dim, 11, seq_length, bio_feature_dim)
model.load_state_dict(torch.load(model_path_large))
model.to(device)
new_model = CNNClassifier(vocab_size, embedding_dim, 11, seq_length, bio_feature_dim)
state_dict = copy.deepcopy(model.state_dict())
state_dict.pop("fc_final.weight")
state_dict.pop("fc_final.bias")
new_state_dict = {k: v.to(device) for k, v in state_dict.items()}
new_model.load_state_dict(new_state_dict, strict=False)
new_model.fc_final = nn.Linear(1024, 11)
new_model.to(device)
new_model.fc_final.to(device)
fine_tune_model(new_model, train_loader, test_loader, num_classes=11)