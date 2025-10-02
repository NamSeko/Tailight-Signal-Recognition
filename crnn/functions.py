import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import yaml
from PIL import Image, ImageOps
from torch.utils import data
import random


def load_config(file_path):
      with open(file_path, "r") as file:
          return yaml.safe_load(file)
      
def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()

def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()

def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Dataset_CRNN_varlen(data.Dataset):
    def __init__(self, lists, labels, set_frame, transform=None, flip_transform=False):
        self.labels = labels
        self.folders, self.video_len = list(zip(*lists))
        self.set_frame = set_frame
        self.transform = transform
        self.flip_transform = flip_transform

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        selected_folder = self.folders[index]
        video_len = self.video_len[index]
        select = np.arange(self.set_frame['begin'], self.set_frame['end'] + 1, self.set_frame['skip'])
        img_size = (self.transform.transforms[0].height, self.transform.transforms[0].width)
        channels = len(self.transform.transforms[-2].mean)

        selected_frames = np.intersect1d(np.arange(1, video_len + 1), select) if self.set_frame['begin'] < video_len else []

        X_padded = torch.zeros((len(select), channels, img_size[0], img_size[1]))
        label = self.labels[index]
        
        flip = False
        if self.flip_transform:
            flip = random.choice([True, False])
            if flip:
                if label == 0:
                    label = 2
                elif label == 2:
                    label = 0

        files = sorted(os.listdir(selected_folder))
        
        torch.manual_seed(random.randint(0, 10000))
        for i, id in enumerate(selected_frames):
            frame = Image.open(os.path.join(selected_folder, files[id-1]))
            if flip:
                frame = ImageOps.mirror(frame)
            
            frame = np.array(frame)
            frame = self.transform(image=frame)['image'] if self.transform is not None else frame
            X_padded[i, :, :, :] = frame
        
        # print(f"Selected folder: {selected_folder}, Label: {label}")        
        y = torch.LongTensor([label])
        video_len = torch.LongTensor([video_len])
        return X_padded, video_len, y
    
#=====================================================================================
class TaillightClassification(nn.Module):
    def __init__(self, num_classes=3):
        super(TaillightClassification, self).__init__()
        self.num_classes = num_classes
        # ResNet18
        # self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # self.feature_dim = self.backbone.fc.in_features
        # self.backbone.fc = nn.Identity()

        # RegNet_Y_1_6GF
        # self.backbone = models.regnet_y_1_6gf(weights=models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V2)
        # self.feature_dim = self.backbone.fc.in_features
        # self.backbone.fc = nn.Identity()
        
        # RegNet_Y_8GF
        self.backbone = models.regnet_y_8gf(weights=models.RegNet_Y_8GF_Weights.IMAGENET1K_V2)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.feature_dim, num_classes)
    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
#=====================================================================================
class ResCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        # resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        clasifier = TaillightClassification(num_classes=3)
        clasifier.load_state_dict(torch.load('/home/namtt/taillight_signal_recognition/classification/models/regnet_y_8gf.pt'))
        
            
        modules = list(clasifier.children())[:-1]
        self.module = nn.Sequential(*modules)
        self.fc1 = nn.Linear(clasifier.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)
        
    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.module(x_3d[:, t, :, :, :])
                x = x.view(x.size(0), -1)

            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        return cnn_embed_seq

class DecoderRNN_varlen(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=50):
        super(DecoderRNN_varlen, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers   
        self.h_RNN = h_RNN                 
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=self.h_RNN_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)
    def forward(self, x_RNN, x_lengths):        
        N, T, n = x_RNN.size()

        for i in range(N):
            if x_lengths[i] < T:
                x_RNN[i, x_lengths[i]:, :] = torch.zeros(T - x_lengths[i], n, dtype=torch.float, device=x_RNN.device)

        x_lengths[x_lengths > T] = T
        lengths_ordered, perm_idx = x_lengths.sort(0, descending=True)

        packed_x_RNN = torch.nn.utils.rnn.pack_padded_sequence(x_RNN[perm_idx], lengths_ordered.cpu(), batch_first=True)
        self.LSTM.flatten_parameters()
        packed_RNN_out, (h_n_sorted, h_c_sorted) = self.LSTM(packed_x_RNN, None)

        RNN_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out, batch_first=True)
        RNN_out = RNN_out.contiguous()
        
        _, unperm_idx = perm_idx.sort(0)
        RNN_out = RNN_out[unperm_idx]
        
        # last_outputs = RNN_out[:, -1, :]
        
        #=============================================================
        lengths_original = lengths_ordered[unperm_idx]
        batch_size = RNN_out.size(0)
        last_outputs = RNN_out[torch.arange(batch_size), lengths_original - 1, :]
        #=============================================================
            
        x = self.fc1(last_outputs)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        return x
