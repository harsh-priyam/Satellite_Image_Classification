import torch 
import torchvision
from PIL import Image
import os 
from torch.utils.data import DataLoader,Dataset,random_split
from torchvision import transforms
import yaml
from pathlib import Path
from src.constants import *

def get_classes():
    with open(CONFIG_PATH,"r") as file:
            config = yaml.load(file,Loader=yaml.SafeLoader)
    path_lib = Path(config['data_preprocessing']['root_dir'])
    classes = {}
    sub_folder_name = [folder for folder in os.listdir(path_lib) if os.path.isdir(os.path.join(path_lib,folder))]
    for i in range(len(sub_folder_name)):
        classes[sub_folder_name[i]] = i 
    return classes


class SatelliteImageDataset(Dataset):
    def __init__(self,config_path:str,transform=None,target_transform = None):

        with open(config_path,"r") as file:
            config = yaml.load(file,Loader=yaml.SafeLoader)

        self.root_dir = Path(config['data_preprocessing']['root_dir'])
        self.transform = transform
        self.target_transform = target_transform
        self.classes = get_classes()
        self.image_path = []
        self.labels = []
    
        for label_name, label_value in self.classes.items():
            label_dir = os.path.join(self.root_dir,label_name)
            for file_name in os.listdir(label_dir):
                if file_name.endswith(('png','jpeg','jpg')):
                    self.image_path.append(os.path.join(label_dir,file_name))
                    self.labels.append(label_value)
        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = index.tolist()
        
        img_path = self.image_path[index]
        label = self.labels[index]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label 
    

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

if __name__ == "__main__":
    config_path = CONFIG_PATH
    param_path = PARAM_PATH
    dataset = SatelliteImageDataset(config_path=config_path,transform=transform)
    dataset_size = dataset.__len__()
