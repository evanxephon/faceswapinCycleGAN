import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from data_augmentation import *
import os

class Dataset(data.Dataset):
    def __init__(self, config):
        
        self.config = config
        
        self.Aimages = []
        self.Bimages = []
        self.transformer_list = []
        
        for imagename in os.listdir(config['imagepath'][0]):
            
            #print(imagename) 
            image = Image.open(os.path.join(config['imagepath'][0] + imagename)).convert('RGB')
            
            image = transforms.Resize((config['resize'],config['resize']), interpolation=Image.BICUBIC)(image)
            
            self.Aimages.append(image)
            
        for imagename in os.listdir(config['imagepath'][1]):
        
            image = Image.open(os.path.join(config['imagepath'][1] + imagename)).convert('RGB')
            
            image = transforms.Resize((config['resize'],config['resize']), interpolation=Image.BICUBIC)(image)
            
            self.Bimages.append(image)
    
        
    def __len__(self):
        return max(len(self.Aimages), len(self.Bimages))
    
    def __getitem__(self, index):

        self.transform = self.get_transform(self.config['augmentation'])
        
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

        warpedA, realA = warp_and_aug(randomAimage, self.config)
        warpedB, realB = warp_and_aug(randomBimage, self.config)
        
        return {'warpedA': warpedA, 'realA': realA, 'warpedB': warpedA, 'realB': realB}
        
    def get_transform(self, config):
        
        if 'rotate_degree' in config.keys():
            self.transformer_list.append(transforms.RandomRotation(np.random.randint(0,config['rotate_degree'])))
            
        if 'flip' in config.keys():
            self.transformer_list.append(transforms.RandomHorizontalFlip())
            
        return transforms.Compose(self.transformer_list)
        
