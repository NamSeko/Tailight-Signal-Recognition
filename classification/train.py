import random
import torch
import torch.nn as nn
import numpy as np
from model import TaillightClassification
from dataloader import get_dataloaders
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # type: ignore
import torchvision


def setseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
setseed(42)

class EarlyStopping:
    def __init__(self, model=None, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.model = model
        self.path = 'regnet_y_8gf.pt'

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            save_model(self.model, path=self.path)
            return False
        elif score < self.best_score - self.delta:
            self.best_score = score
            save_model(self.model, path=self.path)
            self.counter = 0
            return False
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                return True
            return False

def one_epoch(model, traindata, valdata, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    train_loop = tqdm(traindata, desc="Training", unit="batch")
    for images, labels in train_loop:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        train_loop.set_postfix(loss=total_loss / total, accuracy=correct / total)
    avg_loss = total_loss / len(traindata.dataset)
    accuracy = correct / total
    
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        val_loop = tqdm(valdata, desc="Validation", unit="batch")
        for images, labels in val_loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_loop.set_postfix(loss=val_loss / val_total, accuracy=val_correct / val_total)
    avg_val_loss = val_loss / len(valdata.dataset)
    val_accuracy = val_correct / val_total
    
    return avg_loss, accuracy, avg_val_loss, val_accuracy

def save_model(model, path='model.pt'):
    torch.save(model.state_dict(), './models/'+path)
    print(f"Model saved to {path}")

def train(model, train_dataloader, val_dataloader, num_epochs=10, learning_rate=0.001, device='cuda'):
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    early_stopping = EarlyStopping(model=model, patience=5, delta=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scherduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)
    
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_l, train_acc, val_l, val_acc = one_epoch(model, train_dataloader, val_dataloader, criterion, optimizer, device)
        early_stopping_step = early_stopping.step(val_l)
        if early_stopping_step:
            print("Early stopping triggered.")
            break
        train_loss.append(train_l)
        train_accuracy.append(train_acc)
        val_loss.append(val_l)
        val_accuracy.append(val_acc)
        scherduler.step(val_l)
        # scherduler.step()
        print(f"Train Loss: {train_l:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_l:.4f}, Validation Accuracy: {val_acc:.4f}")
        
    print("Training complete.")
    return train_loss, train_accuracy, val_loss, val_accuracy
        
def evaluate(model, test_dataloader, device='cuda'):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    predicted_classes = []
    true_classes = []
    
    with torch.no_grad():
        test_loop = tqdm(test_dataloader, desc="Testing", unit="batch")
        for images, labels in test_loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            predicted_classes.extend(predicted.cpu().numpy())
            true_classes.extend(labels.cpu().numpy())
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            test_loop.set_postfix(loss=test_loss / test_total, accuracy=test_correct / test_total)
    
    avg_test_loss = test_loss / len(test_dataloader.dataset)
    test_accuracy = test_correct / test_total
    
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    return predicted_classes, true_classes, avg_test_loss, test_accuracy

def evaluate_model(model, test_dataloader, device='cuda'):
    model.to(device)
    return evaluate(model, test_dataloader, device)

def plot_results(train_loss, train_accuracy, val_loss, val_accuracy):
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label='Train Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./images/training_results.png')
    plt.close()
    
def plot_confusion_matrix(predicted_classes, true_classes):
    cm = confusion_matrix(true_classes, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['left_signal', 'no_signal', 'right_signal'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig('./images/confusion_matrix.png')
    plt.close()
    
def main():
    train_dir = './data/train'
    val_dir = './data/val'
    test_dir = './data/test'
    
    batch_size = 32
    num_classes = 3
    num_epochs = 50
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train_dir, val_dir, test_dir, batch_size)
    
    # data_iter = iter(train_dataloader)
    # images, labels = next(data_iter)
    # img = torchvision.utils.make_grid(images, nrow=8, normalize=True)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(img.permute(1, 2, 0).cpu())
    # plt.axis('off')
    # plt.savefig('./images/sample_images.png')
    # plt.close()
    
    model = TaillightClassification(num_classes=num_classes)
    
    train_loss, train_acc, val_loss, val_acc = train(model, train_dataloader, val_dataloader, num_epochs=num_epochs, learning_rate=learning_rate, device=device)
    pre_class, tar_class, test_loss, test_acc = evaluate_model(model, test_dataloader, device=device)
    plot_results(train_loss, train_acc, val_loss, val_acc)
    plot_confusion_matrix(pre_class, tar_class)
    
if __name__ == "__main__":
    main()