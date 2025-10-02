import torch
import random
import numpy as np
from model import TaillightClassification
from dataloader import get_dataloaders, get_transforms
from PIL import Image

def setseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
setseed(42)

CLASS_NAMES = {
    0: 'left_signal',
    1: 'no_signal',
    2: 'right_signal'
}

train_transform, test_transform = get_transforms()

model = TaillightClassification(num_classes=3)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.load_state_dict(torch.load('./models/regnet_y_8gf.pt', map_location=device))
img_paths = [
    './data/test/left_signal/image5.jpg',
    './data/test/right_signal/image5.jpg',
    './data/test/no_signal/image5.jpg'
]
for img_path in img_paths:
    
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    img = test_transform(image=img)['image']
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        model.eval()
        output = model(img)
        _, predicted = torch.max(output, 1)
        print(f"Predicted class: {CLASS_NAMES.get(predicted.item())}")
