import numpy as np
import torch.utils.data as data
from PIL import Image
import cv2
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

            image = Image.open(os.path.join(config['imagepath'][0] + imagename)).convert('RGB') 
            
            assert np.all(np.array(image) >= 0), 'need positive matrix'
            
            image = Image.fromarray(np.array(image)[:,:,::-1])
            
            image = transforms.Resize((config['resize'],config['resize']), interpolation=Image.BICUBIC)(image)          
            
            self.Aimages.append(image)
            
        for imagename in os.listdir(config['imagepath'][1]):
        
            image = Image.open(os.path.join(config['imagepath'][0] + imagename)).convert('RGB')
            
            assert np.all(np.array(image) >= 0), 'need positive matrix'
            
            image = Image.fromarray(np.array(image)[:,:,::-1])
            
            image = transforms.Resize((config['resize'],config['resize']), interpolation=Image.BICUBIC)(image)
            
            self.Bimages.append(image)
    
        
    def __len__(self):
        return max(len(self.Aimages), len(self.Bimages))
    
    def __getitem__(self, index):

        self.transform = self.get_transform(self.config['augmentation'])
        
        if index >= len(self.Aimages):
            rawAimage = self.Aimages[index % len(self.Aimages)]
            rawBimage = self.Bimages[index]
            
        elif index >= len(self.Bimages):
            rawBimage = self.Bimages[index % len(self.Bimages)]
            rawAimage = self.Aimages[index]            
        else:
            rawAimage = self.Aimages[index]  
            rawBimage = self.Bimages[index]  
            
        randomAimage = self.transform(rawAimage)
        randomBimage = self.transform(rawBimage)
        
        assert np.all(realA >= 0), 'need positive matrix'
        assert np.all(warpedA >= 0), 'need positive matrix'

        warpedA, realA = warp_and_aug(randomAimage, self.config)
        warpedB, realB = warp_and_aug(randomBimage, self.config)
        
        assert np.all(realA >= 0), 'need positive matrix'
        assert np.all(warpedA >= 0), 'need positive matrix'
        
        warpedA = transforms.functional.to_tensor(warpedA).float()
        realA = transforms.functional.to_tensor(realA).float()
        warpedB = transforms.functional.to_tensor(warpedB).float()
        realB = transforms.functional.to_tensor(realB).float()
        
        return {'warpedA': warpedA, 'realA': realA, 'warpedB': warpedA, 'realB': realB}
        
    def get_transform(self, config):
        
        if 'rotate_degree' in config.keys():
            self.transformer_list.append(transforms.RandomRotation(np.random.randint(0,config['rotate_degree'])))
            
        if 'flip' in config.keys():
            self.transformer_list.append(transforms.RandomHorizontalFlip())
            
        return transforms.Compose(self.transformer_list)
        
