import network
import data_loader
import matplotlib.pyplot as plt

config = {'isTrain': True,
          'loss_weight_config': {'reconstruction_loss': 1,
                                 'adversarial_loss_discriminator': 1,
                                 'adversarial_loss_generator': 1,
                                 'cycle_consistency_loss': 1,
                                },
          'epochs': 1000,
          'cycleepochs': 800,
          'display_interval': 50,
          'augmantation':{'rotate_degree': 5,
                          'flip': True,
                         },
          'imagepath':['./faceA', './faceB'],
          
         }

if __name__ == '__main__':
          
    dataset = data_loader.Dataset()
    model = network.CycleGAN()
    model.train()
    model.cuda()
    model.initialize_weights()

    for epoch in range(config['epochs']):
        if epoch // config['display_interval'] == 0:
            model.displayepoch = True
                    
        if epoch > config['cycleepochs']:
            model.cycle_consistency_loss = True
                    
        for batchdata in dataset:
            
            model.set_input(batchdata)
            model.optimizer_parameters()
                    
        if epoch // config['display_interval'] == 0:
            for image in range(len(model.fakeA)):
                plt.imshow(image)
                plt.show()
                    
        print(f'loss')


                 
            
