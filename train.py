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
          
    
    model = network.CycleGAN(vggface, vggface_ft_pl, config=config)

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
            
            model.train()
            model.cuda()
            model.float()

            model.set_input(batchdata)
            model.optimize_parameter()
                    
        if epoch // config['display_interval'] == 0:
          
            realApic = []
            displayApic = []
            
            realBpic = []
            displayBpic = []
            
            for batch in model.realA:    
                realApic.append(np.squeeze(batch.cpu().detach().numpy()))
           
            realApic = np.concatenate(tuple(realApic), axis=2)
            
            for batch in model.realB:
                realBpic.append(np.squeeze(batch.cpu().detach().numpy()))
           
            realBpic = np.concatenate(tuple(realBpic), axis=2)
            
            for batch in model.displayA:
                displayApic.append(np.squeeze(batch.cpu().detach().numpy()))
           
            displayApic = np.concatenate(tuple(displayApic), axis=2)
            
            for batch in model.displayB:
                displayBpic.append(np.squeeze(batch.cpu().detach().numpy()))
           
            displayBpic = np.concatenate(tuple(displayBpic), axis=2)
            
            pics = [realApic, realBpic, displayApic, displayBpic]
            for i in range(len(pics)):
                assert np.all(pics[i] >= 0), f'{i} need possitive matrix!'
                pics[i] = pics[i][::-1,:,:].copy()
                pics[i] = transforms.functional.to_pil_image(torch.tensor(pics[i]), 'RGB')
                print(type(pics[i]))
                display(pics[i])
                
