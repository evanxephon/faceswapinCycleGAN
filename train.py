import network
import dataset
import os
from IPython import display
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torch
import vggface

config = {'isTrain': True,
          'loss_weight_config': {'reconstruction_loss': 1,
                                 'adversarial_loss_discriminator': 1,
                                 'adversarial_loss_generator': 1,
                                 'cycle_consistency_loss': 1,
                                 'perceptual_loss': 1,
                                },

          'G_lr': 0.0001,
          'D_lr': 0.0002,
          'C_lr': 0.0001,
          'batchsize': 4,
          'resize': 256,
          'epochs': 1000,
          'cycleepochs': 800,
          'display_interval': 50,
          'save_dir': './weights/',
          'save_interval': 100,
          'augmentation':{'rotate_degree': 5,
                          'flip': True,
                         },
          'imagepath':['./faceA/align/', './faceB/align/'],
          
         }

# set global tensor type, must match the model type
torch.set_default_tensor_type(torch.FloatTensor)

# for the debug thing
torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    
    #build model to calculate perceptual loss 
    if not os.path.isdir('./weights'):
        os.mkdir('./weights')
        
    data = dataset.Dataset(config)
    dataloader = DataLoader(data, config['batchsize'], drop_last=True)

    vggface, vggface_ft_pl = vggface.resnet50("resnet50_ft_weight.pkl", num_classes=8631)  # Pretrained weights fc layer has 8631 outputs
          
    model = network.CycleGAN(vggface_ft_pl, config=config)
    
    model.train()
    model.cuda()
    model.float()

    model.initialize_weights()

    for epoch in range(config['epochs']):
        if epoch // config['display_interval'] == 0:
            model.displayepoch = True
            
        if epoch // config['save_interval'] == 0:
            model.save_networks(epoch)
                    
        if epoch > config['cycleepochs']:
            model.cycle_consistency_loss = True
                    
        for batchdata in dataloader:
          
            # need to model.float() every epoch, cuz pytorch reconstruct the grapy every epoch
            model.float()
            model.set_input(batchdata)
            model.optimize_parameter()
                    
        if epoch // config['display_interval'] == 0:
            realA = np.array()
            displayA = np.array()
            
            realBpic = np.array()
            displayBpic = np.array()
            
            for batch in model.realA:
                realApic = realApic.concatenate(np.squeeze(batch), axis=1)
            
            for batch in model.realB:
                realBpic = realBpic.concatenate(np.squeeze(batch), axis=1)
                
            for batch in model.displayA:
                displayApic = displayApic.concatenate(np.squeeze(batch), axis=1)
                
            for batch in model.displayB:
                displayBpic = displayBpic.concatenate(np.squeeze(batch), axis=1)
                
            realApic = cv2.cvtColor(realApic, cv2.COLOR_BGR2RGB)    
            displayApic = cv2.cvtColor(displayApic, cv2.COLOR_BGR2RGB) 
            realBpic = cv2.cvtColor(realBpic, cv2.COLOR_BGR2RGB) 
            displayBpic = cv2.cvtColor(displayBpic, cv2.COLOR_BGR2RGB) 
            
            display(Image.fromarray(realApic))
            display(Image.fromarray(displayApic))
            display(Image.fromarray(realBpic))
            display(Image.fromarray(displayBpic))
            
            # print loss
