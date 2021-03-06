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
          'loss_config':{'pl_on': False,
                         'cyclegan_on': False,
                         'edgeloss_on': False,
                         'eyeloss_on': False,
                         'lr_factor': 1.,
                         'mask_threshold': 0.,
                              },
          'loss_weight_config': {'reconstruction_loss': 1,
                                 'adversarial_loss_discriminator': 0.1,
                                 'adversarial_loss_generator': 0.1,
                                 'cycle_consistency_loss': 0.1,
                                 'perceptual_loss': [0.03, 0.1, 0.3, 0.1],
                                 'mask_loss': 0.01,
                                 'eye_loss': 0.1,
                                 'edge_loss': 0.1,
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
          'imagepath':['./faceA/rgb/', './faceB/rgb/'],
          'eye_mask_dir':['./faceA/eyemask/', './faceB/eyemask/'],
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
                    
        # loss config change during training stage
        if epoch == config['epochs']/5:
            model.loss_['mask_threshold'] = 0.5
            model.loss_config['pl_on'] = True   
                    
        elif epoch == 2*config['epochs']/5:
            model.loss_['mask_threshold'] = 0.5

        elif epoch == config['epochs']/2:
            model.loss_['mask_threshold'] = 0.5
            model.loss_config['lr_factor'] = 0.3
                    
        elif epoch == 2*config['epochs']/3:
            model.loss_['mask_threshold'] = 0.5
            model.loss_config['lr_factor'] = 1
                    
        elif epoch == 8*config['cycleepochs']/10:
            model.loss_config['cyclegan_on'] = True
            model.loss_config['lr_factor'] = 0.3
            model.loss_['mask_threshold'] = 0.5
          
        elif epoch == 9*config['cycleepochs']/10:
            model.loss_config['lr_factor'] = 0.5
          
        if epoch % config['save_interval'] == 0:
            model.save_networks(epoch)
                    
        for batchnum, batchdata in enumerate(dataloader):
          
            model.cuda()
            model.float()
       
            model.set_input(batchdata)
            # model.display_train_data(batchdata)
            model.optimize_parameter()

            # display reconstruction result

            if batchnum == 0:    
                model.display_loss(epoch)
          
                print(f'epoch:{epoch} reconstruction result')
                vis.show_recon_result(model.realA.cpu().detach().numpy(), model.warpedA.cpu().detach().numpy(), 
                                      model.outputA.cpu().detach().numpy(), model.maskA.cpu().detach().numpy())
                vis.show_recon_result(model.realB.cpu().detach().numpy(), model.warpedB.cpu().detach().numpy(), 
                                      model.outputB.cpu().detach().numpy(), model.maskB.cpu().detach().numpy())
                    
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
                     
            vis.show_swap_result(model.realB.cpu().numpy(), model.displayAoutput.cpu().numpy(),
                                 model.displayAmask.cpu().numpy())
            
            vis.show_swap_result(model.realA.cpu().numpy(), model.displayBoutput.cpu().numpy(),
                                 model.displayBmask.cpu().numpy())
            
            
            del model.displayAoutput
            del model.displayBoutput
            del model.displayAmask
            del model.displayBmask
            del model.displayA
            del model.displayB
