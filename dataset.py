import numpy as np
import torch.utils.data as data
from PIL import Image
import cv2
import torchvision.transforms as transforms
from data_augmentation import *
import os
from IPython.display import display

class Dataset(data.Dataset):
    def __init__(self, config, filenames):
        
        self.config = config['augmentation']
        self.eye_mask_dir = config['eye_mask_dir']
        self.Aimages = []
        self.Bimages = []
        self.transformer_list = []
        self.batchsize = config['batchsize']
        self.filenames = filenames
        
        for imagename in os.listdir(config['imagepath'][0]):

            image = Image.open(os.path.join(config['imagepath'][0] + imagename)).convert('RGB') 
            
            if self.eye_mask_dir:
                eyemask = Image.open(os.path.join(config['eye_mask_dir'][0] + imagename))
            
            assert np.all(np.array(image) >= 0), 'need positive matrix'
            
            # turn to bgr mode
            image = Image.fromarray(np.array(image)[:,:,::-1])
            
            image = transforms.Resize((config['resize'],config['resize']), interpolation=Image.BICUBIC)(image) 

            eyemask = transforms.Resize((config['resize'],config['resize']), interpolation=Image.BICUBIC)(eyemask)
            
            imageandmask = Image.fromarray(np.concatenate([np.array(eyemask), np.array(image)], axis=-1))
          
            self.Aimages.append(imageandmask)
            
        for imagename in os.listdir(config['imagepath'][1]):
        
            image = Image.open(os.path.join(config['imagepath'][1] + imagename)).convert('RGB')
            
            if self.eye_mask_dir:
                eyemask = Image.open(os.path.join(config['eye_mask_dir'][1] + imagename))

            assert np.all(np.array(image) >= 0), 'need positive matrix'
            
            # turn to bgr mode
            image = Image.fromarray(np.array(image)[:,:,::-1])
            
            image = transforms.Resize((config['resize'],config['resize']), interpolation=Image.BICUBIC)(image) 

            eyemask = transforms.Resize((config['resize'],config['resize']), interpolation=Image.BICUBIC)(eyemask)
            
            imageandmask = Image.fromarray(np.concatenate([np.array(eyemask), np.array(image)], axis=-1))
            
            self.Bimages.append(imageandmask)
    
        
    def __len__(self):
        return max(len(self.Aimages), len(self.Bimages))
    
    def __getitem__(self, index):

        self.transform = self.get_transform(self.config)
        
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
        
        #display image before data augmentation
#        print('image before augmentation')

#        display(Image.fromarray(np.array(randomAimage)[:,:,::-1]))
#        display(Image.fromarray(np.array(randomBimage)[:,:,::-1]))
            
        assert np.all(np.array(randomAimage) >= 0), 'need positive matrix'
        assert np.all(np.array(randomBimage) >= 0), 'need positive matrix'

        warpedA, realAandeye = warp_and_aug(randomAimage, self.filenames)
        warpedB, realBandeye = warp_and_aug(randomBimage, self.filenames)
        
        warpedA = warpedA[:,:,:3]
        warpedB = warpedB[:,:,:3]
        
        realA = realAandeye[:,:,:3]
        realB = realBandeye[:,:,:3]
        
        eyemaskA = realAandeye[:,:,-1]
        eyemaskB = realBandeye[:,:,-1]
        
        if self.config['motion_blur'] < np.random.randint(0,1):
            warpedA, realA, warpedB, realB = motion_blur([warpedA, realA, warpedB, realB])

        assert np.all(realA >= 0), 'need positive matrix'
        assert np.all(warpedA >= 0), 'need positive matrix'
        
        # the data type should 'uint8' in order to satisfy the to_tensor requirement
        warpedA = transforms.functional.to_tensor(warpedA.astype('uint8')).float()
        realA = transforms.functional.to_tensor(realA.astype('uint8')).float()
        eyemaksA = transforms.functional.to_tensor(eyemaskA.astype('uint8')).float()
        warpedB = transforms.functional.to_tensor(warpedB.astype('uint8')).float()
        realB = transforms.functional.to_tensor(realB.astype('uint8')).float()
        eyemaskB = transforms.functional.to_tensor(eyemaskB.astype('uint8')).float()
       
        
        return {'warpedA': warpedA, 'realA': realA, 'warpedB': warpedB, 'realB': realB, 'eyemaskA': eyemaskA, 'eyemaskB': eyemaskB}
        
    def get_transform(self, config):
        
#         if 'rotate_degree' in config.keys():
#             self.transformer_list.append(transforms.RandomRotation(np.random.randint(0,config['rotate_degree'])))
            
        if 'flip' in config.keys():
            self.transformer_list.append(transforms.RandomHorizontalFlip())
            
        return transforms.Compose(self.transformer_list)
        
