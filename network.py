import torch.nn as nn
from block import *
import loss
from keras.models import Model
import itertools
import os

class Encoder(nn.Module):

    def __init__(self):

        super(Encoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.sablock1 = SABlock(dim_in=256, activation='relu')

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.sablock2 = SABlock(dim_in=512, activation='relu')

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024,
                      kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(1024*4*4, 1024)
        
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 1024*4*4)
        
        self.bn2 = nn.BatchNorm1d(1024*4*4)

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=4096,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4096),
            nn.ReLU(),
        )

        self.upscaleblock = nn.PixelShuffle(2)

    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x, _ = sablock1(x)

        x = self.conv4(x)

        x, _ = self.sablock2(x)
        
        x = x.view([-1,1024*4*4])

        x = fc1(x)
        
        x = bn1(x)

        x = fc2(x)
        
        x = bn2(x)

        x = x.view([-1, 1024, 4, 4])

        x = self.conv6(x)

        x = upscaleblock(x)

        return x


class Decoder(nn.Module):

    def __init__(self, dim_in=512):
        super(Decoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=256 *
                      2*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )

        self.upscaleblock1 = nn.PixelShuffle(2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128*2 *
                      2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )

        self.upscaleblock2 = nn.PixelShuffle(2)

        self.sablock1 = SABlock(dim_in=128, activation=None)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64*2*2,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )

        self.upscaleblock3 = nn.PixelShuffle(2)

        self.resblock = ResidualBlock(dim_in=64)
        
        self.bn = nn.BatchNorm2d(64)

        self.sablock1 = SABlock(dim_in=64, activation=None)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1,
                      kernel_size=5, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3,
                      kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.upscaleblock1(x)

        x = self.conv2(x)
        x = self.upscaleblock2(x)

        x, _ = self.sablock1(x)

        x = self.conv3(x)
        x = self.upscaleblock3(x)

        x = self.resblock(x)
        
        x = self.bn(x)

        x, _ = self.sablock2(x)

        mask = self.conv4(x)

        output = self.conv5(x)

        return output, mask


class Discriminator(nn.Module):

    def __init__(self, dim_in):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=64,
                      kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )

        self.sablock1 = SABlock(128, activation=None)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )

        self.sablock2 = SABlock(256, activation=None)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1,
                      kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)

        x, _ = self.sablock1(x)
        x = self.conv3(x)

        x, _ = self.sablock2(x)
        x = self.conv4(x)

        return x

    
class CycleGAN(nn.Module):
    
    def __init__(self, vggface, config):
        
        super(CycleGAN, self).__init__()
        
        self.Encoder = Encoder()
        self.DecoderA = Decoder()
        self.DecoderB = Decoder()
        
        self.model_names = ['Encoder', 'DecoderA', 'DecoderB', 'DiscriminatorA', 'DiscriminatorB']
        self.isTrain = config['isTrain']
        self.cycle_consistency_loss = False
        self.loss_weight_config = config['loss_weight_config']
        self.vggface_feats = vggface
        self.optimizers = []
        self.save_dir = config['save_dir']
        self.loss_value = {}
          
        self.display_epoch = False
        
        if self.isTrain:
            self.DiscriminatorA = Discriminator(3)
            self.DiscriminatorB = Discriminator(3)
            
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.Encoder.parameters(), self.DecoderA.parameters(),
                                                                self.DecoderB.parameters()), lr=config['G_lr'])#betas=(opt.beta1, 0.999)        
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.DiscriminatorA.parameters(), self.DiscriminatorB.parameters()),
                                                                 lr=config['D_lr'])#, betas=(opt.beta1, 0.999)) 
            self.optimizer_Cycle = torch.optim.Adam(itertools.chain(self.Encoder.parameters(), self.DecoderA.parameters(),
                                                    self.DecoderB.parameters(),self.DiscriminatorA.parameters(),
                                                    self.DiscriminatorB.parameters()), lr=config['C_lr'])#, betas=(opt.beta1, 0.999))
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_Cycle)
        
    def set_input(self, input):
        
        self.warpedA = input['warpedA']
        self.warpedB = input['warpedB']
        self.realA = input['realA']
        self.realB = input['realB']
        
    def forward(self):
        
        if self.display_epoch == True:
            self.displayAoutput, self.displayBmask = self.DecoderA(self.Encoder(self.realB))
            self.displayBoutput, self.displayBmask = self.DecoderB(self.Encoder(self.warpedB))

            self.displayA = self.displayAmask * self.displayAoutput + (1 - self.displayAmask) * self.realB
            self.displayB = self.displayBmask * self.displayBoutput + (1 - self.displayBmask) * self.realA 
              
        if not self.isTrain or self.cycle_consistency_loss:
            self.warpedA = self.realB
            self.warpedB = self.realA
            
        # mask(Alpha) output and BGR output
        self.outputA, self.maskA = self.DecoderA(self.Encoder(self.warpedA))
        
        self.outputB, self.maskB = self.DecoderB(self.Encoder(self.warpedB))
        
        # combine mask and output to get fake result
        self.fakeA = self.maskA * self.outputA + (1 - self.maskA) * self.warpedA
        self.fakeB = self.maskB * self.outputB + (1 - self.maskB) * self.warpedB  
        
        if self.isTrain:
            self.fakeApred = self.Discriminator(self.fakeA)
            self.fakeBpred = self.Discriminator(self.fakeB)
            self.realApred = self.Discriminator(self.realA)
            self.realBpred = self.Discriminator(self.realB)
        
        
        if self.cycle_consistency_loss:
            
            self.cycleA = self.DecoderA(self.Encoder(self.outputB))
            self.cycleB = self.DecoderB(self.Encoder(self.outputA))
            
    def backward_D_A(self):
        
        loss_D_A = loss.adversarial_loss_discriminator(self.fakeA, self.realA, 'L2', loss_weight_config)
        self.loss_value['loss_D_A'] = loss_D_A.detach()
        loss_D_A.backward()
        
    def backward_D_B(self):
        
        loss_D_B = loss.adversarial_loss_discriminator(self.fakeB, self.realB, 'L2', loss_weight_config)
        self.loss_value['loss_D_B'] = loss_D_A.detach()
        loss_D_B.backward()
      
    def backward_G_A(self):
        
        loss_G_adversarial_loss = loss.adversarial_loss_generator(self.fakeA, 'L2', loss_weight_config)
        self.loss_value['loss_G_adversarial_loss_A'] = loss_G_adversarial_loss.detach()
        
        loss_G_reconstruction_loss = loss.reconstruction_loss(self.fakeA, self.realA, 'L2', loss_weight_config)
        self.loss_value['loss_G_reconstruction_loss_A'] = loss_G_reconstruction_loss.detach()
        
        loss_G_perceptual_loss = loss.perceptual_loss(self.realA, self.fakeA, self.vggface_feats, 'L2', loss_weight_config)
        self.loss_value['loss_G_perceptual_loss_A'] = loss_G_perceptual_loss.detach()
        
        loss_G_A = loss_G_adversarial_loss + loss_G_reconstruction_loss + loss_G_perceptual_loss
        loss_G_A.backward()
        
    def backward_G_B(self):
        
        loss_G_adversarial_loss = loss.adversarial_loss_generator(self.fakeA, 'L2', loss_weight_config)
        self.loss_value['loss_G_adversarial_loss_A'] = loss_G_adversarial_loss.detach()
        
        loss_G_reconstruction_loss = loss.reconstruction_loss(self.fakeA, self.realA, 'L2', loss_weight_config)
        self.loss_value['loss_G_reconstruction_loss_A'] = loss_G_reconstruction_loss.detach()
        
        loss_G_perceptual_loss = loss.perceptual_loss(self.realA, self.fakeA, self.vggface_feats, 'L2', loss_weight_config)
        self.loss_value['loss_G_perceptual_loss_A'] = loss_G_perceptual_loss.detach()
        
        loss_G_B = loss_G_adversarial_loss + loss_G_reconstruction_loss + loss_G_perceptual_loss
        loss_G_B.backward()
        
    def backward_Cycle_A(self):
        
        loss_Cycle_A = loss.cycle_consistency_loss(self.realA, self.cycleA, 'L2', loss_weight_config)
        self.loss_value['loss_Cycle_A'] = loss_loss_Cycle_A.detach()
        
        loss_Cycle_A.backward()
        
    def backward_Cycle_B(self):
        
        loss_Cycle_B = loss.cycle_consistency_loss(self.realB, self.cycleB, 'L2', loss_weight_config)
        self.loss_value['loss_Cycle_B'] = loss_loss_Cycle_B.detach()
        
    def optimize_parameter(self):
    
        self.forward()
        
        self.set_requires_grad([self.Encoder, self.DecoderA, self.DecoderB], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
        
        if self.cycle_consistency_loss:
            self.backward_Cycle_A()
            self.backward_Cycle_B()
            self.optimizer_Cycle.step()
        
        else:
            self.set_requires_grad([self.DiscriminatorA, self.DiscriminatorB], False)
            self.optimizer_G.zero_grad()

            self.backward_G_A()
            self.backward_G_B()
            self.optimizer_G.step()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # print(m.weight.data.type())
                # input()
                # m.weight.data.fill_(1.0)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                #print(m.weight)
                
    def save_networks(self, epoch):
        
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = f'{epoch}_net_{name}.pth'
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, name)

                if torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                else:
                    torch.save(net.cpu().state_dict(), save_path)
                    
    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                
                net = getattr(self, name)
                
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                    
                print('loading the model from %s' % save_dir)
                state_dict = torch.load(load_path)
                
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                    
                net.load_state_dict(state_dict)
    
def vggface_for_pl(vggface_keras, **loss_weight_config):
    
    vggface_keras.trainable = False
    
    out_size112 = vggface_keras.layers[15].output
    out_size55 = vggface_keras.layers[35].output
    out_size28 = vggface_keras.layers[77].output
    out_size7 = vggface_keras.layers[-3].output
    
    vggface_feats = Model(vggface_keras.input, [out_size112, out_size55, out_size28, out_size7])
    vggface_feats.trainable = False
    
    return vggface_feats
