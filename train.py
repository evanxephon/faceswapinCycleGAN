import network
import dataset
import os
import cv2
import numpy as np
from torch.utils.data import DataLoader
import torch
import vggface
from glob import glob
import imp
import visualization as vis

config = {'isTrain': True,
          'loss_weight_config': {'reconstruction_loss': 1,
                                 'adversarial_loss_discriminator': 0.1,
                                 'adversarial_loss_generator': 0.1,
                                 'cycle_consistency_loss': 0.1,
                                 'perceptual_loss': [0.03, 0.1, 0.3, 0.1],
                                },

          'G_lr': 0.0001,
          'D_lr': 0.0002,
          'C_lr': 0.0001,
          'batchsize': 8,
          'resize': 256,
          'epochs': 1000,
          'cycleepochs': 800,
          'display_interval': 1,
          'save_dir': './weights/',
          'save_interval': 100,
          'augmentation':{'rotate_degree': 5,
                          'flip': True,
                          'motion_blur': 0.6,
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

    trainsetAfilenames = glob(config['imagepath'][0] + '*.*')
    trainsetBfilenames = glob(config['imagepath'][1] + '*.*')

    filenames = trainsetAfilenames + trainsetBfilenames
          
    data = dataset.Dataset(config, filenames)
    dataloader = DataLoader(data, config['batchsize'], drop_last=True)
   
    vggface, vggface_ft_pl = vggface.resnet50("resnet50_ft_weight.pkl", num_classes=8631)  # Pretrained weights fc layer has 8631 outputs
          
    model = network.CycleGAN(vggface, vggface_ft_pl, config=config)
    
    model.train()

#    model.initialize_weights()

    for epoch in range(config['epochs']):

        if epoch % config['save_interval'] == 0:
            model.save_networks(epoch)
                    
        if epoch >= config['cycleepochs']:
            model.cycle_consistency_loss = True
                    
        for batchnum, batchdata in enumerate(dataloader):
          
            model.cuda()
            model.float()
       
            model.set_input(batchdata)
            # model.display_train_data(batchdata)
            model.optimize_parameter()
            model.display_loss(epoch)
            
            # display reconstruction result

            if batchnum == 0:
                    
                print(f'epoch:{epoch} reconstruction result')
              
                vis.show_recon_result(model.realA.cpu().detach().numpy(), model.warpedA.cpu().detach().numpy(), 
                                      model.fakeA.cpu().detach().numpy(), model.maskA.cpu().detach().numpy())

                vis.show_recon_result(model.realB.cpu().detach().numpy(), model.warpedB.cpu().detach().numpy(), 
                                      model.fakeB.cpu().detach().numpy(), model.maskB.cpu().detach().numpy())
                    
            # del mannully
            
            del model.realA
            del model.realB
            del model.warpedA
            del model.warpedB
            del model.outputA
            del model.outputB
            del model.maskA
            del model.maskB
            del model.fakeA
            del model.fakeB
            del model.fakeApred
            del model.fakeBpred
            del model.realApred             
            del model.realBpred
            if model.cycle_consistency_loss:
                del model.cycleA
                del model.cycleB
                
        if epoch % config['display_interval'] == 0:
          
            batchdata = iter(dataloader).next()
            model.set_input(batchdata)
            model.display_forward()
            
            print(f'display result epoch: {epoch}')
                     
            vis.show_swap_result(model.realB.cpu().numpy(), model.displayA.cpu().numpy(),
                                 model.displayAmask.cpu().numpy())
            
            vis.show_swap_result(model.realA.cpu().numpy(), model.displayB.cpu().numpy(),
                                 model.displayBmask.cpu().numpy())
            
            
            del model.displayAoutput
            del model.displayBoutput
            del model.displayAmask
            del model.displayBmask
            del model.displayA
            del model.displayB
