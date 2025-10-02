from dataloader import get_transforms, TailLightDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import cv2
import numpy as np
from model import TaillightClassification
from tqdm import tqdm
import os

def setseed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
setseed(42)

CLASS_MAPPING = {
    0: 'left_signal',
    1: 'no_signal',
    2: 'right_signal'
}

test_transform = get_transforms()[1]
def check_false_positive(model, test_dir, device='cuda'):
    model.to(device)
    model.eval()
    dataset = TailLightDataset(test_dir, transform=test_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    criterion = nn.CrossEntropyLoss()
    false_predicted_labels = []
    false_predicted_images = []
    true_labels = []
    with torch.no_grad():
        loop = tqdm(dataloader, desc="Checking false positives", unit="batch")
        for img, label in loop:
            img = img.to(device)
            label = label.to(device)
            output = model(img)
            loss = criterion(output, label)
            _, predicted = torch.max(output, 1)
            if predicted.item() != label.item():
                false_predicted_images.append(img.cpu())
                true_labels.append(label.item())
                false_predicted_labels.append(predicted.item())
    return false_predicted_images, false_predicted_labels, true_labels

def unnormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # x = x * std + mean
    return tensor

def main():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # Load model
    model = TaillightClassification(num_classes=3)
    model.load_state_dict(torch.load('./models/regnet_y_8gf.pt', map_location=device))
    
    test_dir = './data/test'
    count = 1
    path_false = f'./images/false{count}'
    while os.path.exists(path_false):
        count += 1
        path_false = f'./images/false{count}'
    os.makedirs(path_false, exist_ok=True)
    false_predict_images, false_predict_labels, true_labels = check_false_positive(model, test_dir, device)
    
    print(f"Number of false predict images: {len(false_predict_images)}")
    for i, img in enumerate(false_predict_images):
        # Dùng cho ImageNet
        pre_label = CLASS_MAPPING.get(false_predict_labels[i])
        tar_label = CLASS_MAPPING.get(true_labels[i])
        unnorm_img = unnormalize(img.clone(), mean, std)
        img = unnorm_img.squeeze().permute(1, 2, 0).numpy()  # Convert to HWC format
        img = np.clip(img, 0, 1)
        img = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{path_false}/pre_{pre_label}.tar_{tar_label}{i}.jpg", img)
        print(f"False positive image {i} saved with predict label {pre_label} and true label {tar_label}")
        
if __name__ == "__main__":
    main()