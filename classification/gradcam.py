import cv2
import random
import numpy as np
from PIL import Image
import torch
from pytorch_grad_cam import GradCAM # type: ignore
from pytorch_grad_cam.utils.image import show_cam_on_image # type: ignore
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget # type: ignore
import os 
from model import TaillightClassification
from dataloader import get_dataloaders, get_transforms

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

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = TaillightClassification(num_classes=3)
# model.load_state_dict(torch.load('./models/model_mobilenet.pt', map_location=device))
model.load_state_dict(torch.load('./models/regnet_y_8gf.pt', map_location=device))
model.eval()

# Select target layer
# target_layers = [model.backbone.layer4[-1]] # ResNet18
# target_layers = [model.backbone.features[-1]]  # MobileNetV3 Small
target_layers = [model.backbone.trunk_output]  # RegNet_Y_1_6GF

def gradcam_plot(img_path, img_name):
    img = Image.open(img_path+'/'+img_name).convert('RGB')
    img_pd = np.array(img)
    input_tensor = test_transform(image=img_pd)['image'].unsqueeze(0) 

    img_float = np.array(img) / 255.0
    img_float = cv2.resize(img_float, (224, 224))
    # Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    output = model(input_tensor)  # Forward pass to get model output
    _, predicted_class = torch.max(output, 1)
    print(f"Predicted class: {CLASS_NAMES.get(predicted_class.item())}")
    targets = [ClassifierOutputTarget(predicted_class.item())]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    # Chuyển về kích thước gốc
    visualization = cv2.resize(visualization, (img.size[0], img.size[1]))

    # path_save = './images/gradcam'
    path_save = './images/grad_false'
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    # Save
    cv2.imwrite(f"{path_save}/gradcam_{img_name}", visualization * 255)
    # cv2.imwrite(f"{path_save}/gradcam_{img_name}", visualization * 255)
    print(f"Grad-CAM saved for {img_name}")
    
def main():
    # test_dir = './images'
    # images = [
    #     'image_left.png',
    #     'image_no.png',
    #     'no_1.jpg',
    #     'no_2.jpg',
    #     'no_3.jpg',
    #     'no_4.jpg',
    # ]
    test_dir = './images/false1'
    images = os.listdir(test_dir)
    for img_name in images:
        gradcam_plot(test_dir, img_name)

if __name__ == "__main__":
    main()