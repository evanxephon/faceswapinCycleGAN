import network
import dataset
#import matplotlib.pyplot as plt
from keras_vggface.vggface import VGGFace
import os
from IPython import display
import cv2
import numpy as np
from PIL import Image
import torch.utils.data.DataLoader


config = {'isTrain': True,
          'loss_weight_config': {'reconstruction_loss': 1,
                                 'adversarial_loss_discriminator': 1,
                                 'adversarial_loss_generator': 1,
                                 'cycle_consistency_loss': 1,
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

if __name__ == '__main__':
    
    #build model to calculate perceptual loss 
    if not os.path.isdir('./weights'):
        os.mkdir('./weights')
    
    vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))
    vggface_feats = network.vggface_for_pl(vggface, loss_weight_config=config['loss_weight_config'])      
        
    data = dataset.Dataset(config)
    dataloader = DataLoader(data, config['batchsize'])
    model = network.CycleGAN(vggface_feats, config=config)
    model.train()
    model.cuda()
    model.initialize_weights()

    for epoch in range(config['epochs']):
        if epoch // config['display_interval'] == 0:
            model.displayepoch = True
            
        if epoch // config['save_interval'] == 0:
            model.save_networks(epoch)
                    
        if epoch > config['cycleepochs']:
            model.cycle_consistency_loss = True
                    
        for batchdata in dataloader:
            
            model.set_input(batchdata)
            model.optimizer_parameters()
                    
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
                    
        print(f'loss')


            
