import network
import data_loader
#import matplotlib.pyplot as plt
from keras_vggface.vggface import VGGFace
import os

config = {'isTrain': True,
          'loss_weight_config': {'reconstruction_loss': 1,
                                 'adversarial_loss_discriminator': 1,
                                 'adversarial_loss_generator': 1,
                                 'cycle_consistency_loss': 1,
                                },

          'G_lr': 0.0001,
          'D_lr': 0.0002,
          'C_lr': 0.0001,
          'resize': 256,
          'epochs': 1000,
          'cycleepochs': 800,
          'display_interval': 50,
          'save_dir': './weights/'
          'save_interval': 100,
          'augmantation':{'rotate_degree': 5,
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
        
    dataset = data_loader.Dataset(config)
    model = network.CycleGAN(vggface_feats, config=config)
    model.train()
    model.cuda()
    model.initialize_weights()

    for epoch in range(config['epochs']):
        if epoch // config['display_interval'] == 0:
            model.displayepoch = True
            
        if epoch // config['save_interval'] == 0:
            model.save_network(epoch)
                    
        if epoch > config['cycleepochs']:
            model.cycle_consistency_loss = True
                    
        for batchdata in dataset:
            
            model.set_input(batchdata)
            model.optimizer_parameters()
                    
        '''if epoch // config['display_interval'] == 0:
            for batch in range(len(model.displayA)):
                plt.imshow(model.displayA[batch])
                plt.show()'''
                    
        print(f'loss')


            
