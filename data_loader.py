import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from data_augmentation import *


class dataset(data.Dataset):
    def __init__(self, config):
        
        self.config = config
        
        self.Aimages = []
        self.Bimages = []
        
        self.transform = get_transform(config['augmentation'])
        
        for imagename in os.listdir(config['imagepath'][0]):
        
            image = Image.open(imagename).convert('RGB')
            
            image = transforms.Resize((config['resize'],config['resize']), method=Image.BICUBIC)(image)
            
            self.Aimages.append(image)
            
        for imagename in os.listdir(config['imagepath'][1]):
        
            image = Image.open(imagename).convert('RGB')
            
            image = transforms.Resize((config['resize'],config['resize']), method=Image.BICUBIC)(image)
            
            self.Bimages.append(image)
    
        
    def __len__(self):
        return max(len(self.Aimages), len(self.Bimages))
    
    def __getitem__(self, index)
        
        if index > len(self.Aimages):
            rawAimage = self.Aimages[index % len(self.Aimages)]
            rawBimage = self.Bimages[index]
            
        elif index > len(self.Bimages):
            rawBimage = self.Bimages[index % len(self.Aimages)]
            rawAimage = self.Aimages[index]            
        else:
            rawAimage = self.Aimages[index]  
            rawBimage = self.Bimages[index]  
            
        randomAimage = self.transform(rawAimage)
        randomBimage = self.transform(rawBimage)

        warpedA, realA = random_warp(image, self.config)
        warpedB, realB = random_warp(image, self.config)
        
        return {'warpedA': warpedA, 'realA': realA, 'warpedB': warpedA, 'realB': realB}
        
    def get_transform(config):
        
        if 'rotate_degree' in config.keys():
            transformer_list.append(transforms.RandomRotation(config['rotate_degree']))
            
        if 'flip' in config.keys():
            transformer_list.append(transforms.RandomHorizontalFlip())
            
        return transforms.Compose(transformer_list)
        
