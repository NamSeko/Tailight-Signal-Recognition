import os

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from functions import *
from sklearn.metrics import accuracy_score # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # type: ignore
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import albumentations as A # type: ignore
from albumentations.pytorch import ToTensorV2 # type: ignore

# Load configuration
config = load_config("./config.yaml")
data_path = str(config["data_path"])
action_names = list(config["action_names"])
save_model_path = str(config["save_model_path"])
check_mkdir(save_model_path)

# Training parameters
k = len(action_names)                       
epochs = int(config["epochs"])              
patience = int(config["patience"])
batch_size = int(config["batch_size"])
num_workers = int(config["num_workers"])
lr_patience = int(config["lr_patience"])
log_interval = int(config["log_interval"])
select_frame = dict(config["select_frame"])
learning_rate = float(config["learning_rate"])

print(f'Data path:', data_path)
print(f'Save model path:', save_model_path)
print(f'Action names:', action_names)
print(f'Number of classes:', k)
print(f'Epochs:', epochs)
print(f'Patience:', patience)
print(f'Batch size:', batch_size)
print(f'Num workers:', num_workers)
print(f'LR patience:', lr_patience)
print(f'Log interval:', log_interval)
print(f'Learning rate:', learning_rate)
print(f'Select frame:', select_frame, '\n')

# Set random seed
set_seed(42)

# EncoderCNN architecture
# CNN_fc_hidden1, CNN_fc_hidden2 = 512, 256
CNN_fc_hidden1, CNN_fc_hidden2 = 768, 512

CNN_embed_dim = 256   # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.2       # dropout probability

# DecoderRNN architecture
RNN_hidden_layers = 3
RNN_hidden_nodes = 256
RNN_FC_dim = 128

def train(log_interval, model, device, train_loader, optimizer, epoch):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    epoch_loss, all_y, all_y_pred = 0, [], []
    N_count = 0
    train_loop = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}", unit="batch")
    # for batch_idx, (X, X_lengths, y) in enumerate(train_loop):
    for (X, X_lengths, y) in train_loop:
        X, X_lengths, y = X.to(device), X_lengths.to(device).view(-1, ), y.to(device).view(-1, )

        N_count += X.size(0)

        optimizer.zero_grad()
        output = rnn_decoder(cnn_encoder(X), X_lengths)

        loss = F.cross_entropy(output, y, label_smoothing=0.1)
        # epoch_loss += F.cross_entropy(output, y, reduction='sum').item()
        epoch_loss += loss.item()

        y_pred = torch.max(output, 1)[1]

        all_y.extend(y)
        all_y_pred.extend(y_pred)

        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())

        loss.backward()
        optimizer.step()

        train_loop.set_postfix(loss=loss.item(), accuracy=step_score*100)
        # if (batch_idx + 1) % log_interval == 0:
        #     print('Train Epoch: {:>4} [{:>4}/{} ({:>3.0f}%)]  Loss: {:<10.6f} Accu: {:>6.2f}%'.format(
        #         epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))
    
    epoch_loss /= len(train_loader)

    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    epoch_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())
    return epoch_loss, epoch_score

def validation(model, device, val_loader):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y, all_y_pred = [], []
    test_loop = tqdm(val_loader, desc="Validation", unit="batch")
    with torch.no_grad():
        for X, X_lengths, y in test_loop:
            X, X_lengths, y = X.to(device), X_lengths.to(device).view(-1, ), y.to(device).view(-1, )

            output = rnn_decoder(cnn_encoder(X), X_lengths)

            loss = F.cross_entropy(output, y, reduction='sum', label_smoothing=0.1)
            test_loss += loss.item()                 
            y_pred = output.max(1, keepdim=True)[1]  

            all_y.extend(y)
            all_y_pred.extend(y_pred)

            test_loop.set_postfix(loss=loss.item())
    test_loss /= len(val_loader.dataset)

    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    print('\nValidation set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))
    return test_loss, test_score

def test(model, device, test_loader):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    all_y, all_y_pred = [], []
    test_loop = tqdm(test_loader, desc="Test", unit="batch")
    with torch.no_grad():
        for X, X_lengths, y in test_loop:
            X, X_lengths, y = X.to(device), X_lengths.to(device).view(-1, ), y.to(device).view(-1, )

            output = rnn_decoder(cnn_encoder(X), X_lengths)

            y_pred = output.max(1, keepdim=True)[1]  

            all_y.extend(y)
            all_y_pred.extend(y_pred)

            test_loop.set_postfix()

    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    print('\nTest set ({:d} samples): Accuracy: {:.2f}%\n'.format(len(all_y), 100* test_score))

use_cuda = torch.cuda.is_available()                   
device = torch.device("cuda" if use_cuda else "cpu")

params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

le = LabelEncoder()
le.fit(action_names)

action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)

actions = []
fnames = os.listdir(data_path)

actions = []
all_names = []
all_length = []
for action in os.listdir(data_path):
    action_folder = os.path.join(data_path, action)
    for folder in os.listdir(action_folder):
        folder_path = os.path.join(action_folder, folder)
        actions.append(action)
        all_names.append(folder_path)
        all_length.append(len(os.listdir(folder_path)))

all_X_list = list(zip(all_names, all_length))
all_y_list = labels2cat(le, actions)

# train, test split
train_list, val_list, train_label, val_label = train_test_split(all_X_list, all_y_list, test_size=0.3, random_state=42, shuffle=True, stratify=all_y_list)
val_list, test_list, val_label, test_label = train_test_split(val_list, val_label, test_size=0.3, random_state=42, shuffle=True, stratify=val_label)
# fit training
# train_list  = all_X_list
# train_label = all_y_list
# test_list  = train_list.copy()
# test_label = train_label.copy()

#==========================================================================================================
train_transform = A.Compose([
    # A.Resize(res_size, res_size),
    # A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
    # A.RandomShadow(shadow_dimension=5, p=0.5),
    # A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.5),
    # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    # A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    # ToTensorV2()
    
    A.Resize(res_size, res_size),
    # tăng tương phản / làm nét
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3),
        A.Sharpen(alpha=(0.2,0.5), lightness=(0.5,1.0), p=0.3),
    ], p=0.8),
    
    # Giả lập ảnh bé / mất nét
    A.OneOf([
        A.MotionBlur(blur_limit=3, p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.ImageCompression(p=0.4),
    ], p=0.5),
    
    # occlusion + noise (bắt model học với mất mát thông tin)
    A.CoarseDropout(p=0.3),
    A.GaussNoise(p=0.3),
    
    A.RandomShadow(shadow_dimension=5, p=0.5),
    A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
val_transform = A.Compose([
    A.Resize(res_size, res_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

train_set, valid_set, test_set = Dataset_CRNN_varlen(train_list, train_label, select_frame, transform=train_transform, flip_transform=True), \
                        Dataset_CRNN_varlen(val_list, val_label, select_frame, transform=val_transform), \
                        Dataset_CRNN_varlen(test_list, test_label, select_frame, transform=val_transform)
#==========================================================================================================

train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)
test_loader = data.DataLoader(test_set, **params)

data_iter = iter(train_loader)
X, X_lengths, y = next(data_iter)
print(f"X shape: {X.shape}, X_lengths shape: {X_lengths.shape}, y shape: {y.shape}")
# Lưu ảnh sample
if not os.path.exists('./images/sample'):
    os.makedirs('./images/sample')
for i in range(X.shape[0]):
    img = X[i].permute(1, 2, 0).cpu().numpy()  # Chuyển đổi về định dạng HWC
    img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255  # Chuyển đổi về giá trị pixel
    img = img.astype(np.uint8)  # Chuyển đổi kiểu dữ liệu
    cv2.imwrite(f'./images/sample/sample_{i}.png', img)
    break

# Dừng tại đây để kiểm tra dữ liệu
exit()

# Create model
cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder = DecoderRNN_varlen(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

# Combine all EncoderCNN + DecoderRNN parameters
print("Using", torch.cuda.device_count(), "GPU!\n")
if torch.cuda.device_count() > 1:
    # Parallelize model to multiple GPUs
    cnn_encoder = nn.DataParallel(cnn_encoder)
    rnn_decoder = nn.DataParallel(rnn_decoder)
    crnn_params = list(cnn_encoder.module.fc1.parameters()) + list(cnn_encoder.module.bn1.parameters()) + \
                  list(cnn_encoder.module.fc2.parameters()) + list(cnn_encoder.module.bn2.parameters()) + \
                  list(cnn_encoder.module.fc3.parameters()) + list(rnn_decoder.parameters())

elif torch.cuda.device_count() == 1:
    crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
                  list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
                  list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters())

optimizer = torch.optim.AdamW(crnn_params, lr=learning_rate, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=lr_patience, min_lr=1e-10)

epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []

# start training
best_score = 0.0
best_epoch = -1
early_stop_counter = 0
for epoch in range(epochs):
    epoch_train_loss, epoch_train_score = train(log_interval, [cnn_encoder, rnn_decoder], device, train_loader, optimizer, epoch)
    epoch_test_loss, epoch_test_score = validation([cnn_encoder, rnn_decoder], device, valid_loader)
    scheduler.step(epoch_test_loss)

    epoch_train_losses.append(epoch_train_loss)
    epoch_train_scores.append(epoch_train_score)
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_scores.append(epoch_test_score)

    A = np.array(epoch_train_losses)
    B = np.array(epoch_train_scores)
    C = np.array(epoch_test_losses)
    D = np.array(epoch_test_scores)
    np.save(os.path.join(save_model_path,'./training_loss.npy'), A)
    np.save(os.path.join(save_model_path,'./training_score.npy'), B)
    np.save(os.path.join(save_model_path,'./test_loss.npy'), C)
    np.save(os.path.join(save_model_path,'./test_score.npy'), D)

    # Save the last model
    torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_last.pth'))
    torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_last.pth'))
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_last.pth'))
    print(f"Epoch {epoch + 1}: New last model saved!")

    # Save the best model
    if epoch_test_score > best_score:
        best_score = epoch_test_score
        best_epoch = epoch
        early_stop_counter = 0

        torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_best.pth'))
        torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_best.pth'))
        torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_best.pth'))
        print(f"Epoch {epoch + 1}: New best model saved!\n")
    else:
        early_stop_counter += 1
        print(f"Epoch {epoch + 1}: No improvement. Early stop counter = {early_stop_counter}/{patience}\n")

    # Early stopping
    if early_stop_counter == patience:
        print(f"Epoch {epoch + 1}: Early stopping. Best epoch was {best_epoch + 1} with score {best_score:.2f}")
        break
    
test_score = test([cnn_encoder, rnn_decoder], device, test_loader)