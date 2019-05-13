import torch.nn as nn
from block import *
import loss

class Encoder(nn.Module):

    def __init__(self):

        super(Encoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
        )

        self.sablock1 = SABlock(dim_in=256, activation='relu')

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
        )

        self.sablock2 = SABlock(dim_in=512, activation='relu')

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024,
                      kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(1024*4*4, 1024)

        self.fc2 = nn.Linear(1024, 1024*4*4)

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=4096,
                      kernel_size=3, stride=1, padding=1),
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

        x = fc1(x)

        x = fc2(x)

        x = x.view(-1, 1024, 4, 4)

        x = self.conv6(x)

        x = upscaleblock(x)

        return x


class Decoder(nn.module):

    def __init__(self, dim_in=512):
        super(Decoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=256 *
                      2*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

        self.upscaleblock1 = nn.PixelShuffle(2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128*2 *
                      2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

        self.upscaleblock2 = nn.PixelShuffle(2)

        self.sablock1 = SABlock(dim_in=128, activation=None)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64*2*2,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

        self.upscaleblock3 = nn.PixelShuffle(2)

        self.resblock = ResidualBlock(dim_in=64)

        self.sablock1 = SABlock(dim_in=64, activation=None)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1,
                      kernel_size=5, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3,
                      kernel_size=5, stride=1, padding=0),
            nn.tanh(),
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
            nn.LeakyReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
        )

        self.sablock1 = SABlock(128, activation=None)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
        )

        self.sablock2 = SABlock(256, activation=None)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1,
                      kernel_size=5, stride=1, padding=0),
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
    
    def __init__(self, config):
        
        super(CycleGAN, self).__init__()
        
        self.Encoder = Encoder()
        self.DecoderA = Decoder()
        self.DecoderB = Decoder()
        
        self.isTrain = config['isTrain']
        self.cycle_consistency_loss = False
        self.loss_weight_config = config['loss_weight_config']
          
        
        self.display_epoch = False
        
        if self.isTrain:
            self.DiscriminatorA = Discriminator()
            self.DiscriminatorB = Discriminator()
            
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.Encoder.parameters(), self.DecoderA.parameters(),
                                                                self.DecoderB.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))        
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.DiscriminatorA.parameters(), self.DiscriminatorB.parameters()),
                                                                 lr=opt.lr, betas=(opt.beta1, 0.999)) 
            self.optimizer_Cycle = torch.optim.Adam(itertools.chain(self.Encoder.parameters(), self.DecoderA.parameters(),
                                                    self.DecoderB.parameters(),self.DiscriminatorA.parameters(),
                                                    self.DiscriminatorB.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            
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
	    self.displayBmask = self.DecoderA(self.Encoder(self.realB))[:,:,:,1]
            self.displayAoutput = self.DecoderA(self.Encoder(self.realA))[:,:,:,1:]

            self.displayBmask = self.DecoderB(self.Encoder(self.warpedB))[:,:,:,1]
            self.displayBoutput = self.DecoderB(self.Encoder(self.warpedB))[:,:,:,1:]

            self.displayA = self.displayAmask * self.displayAoutput + (1 - self.displayAmask) * self.realB
            self.displayB = self.displayBmask * self.displayBoutput + (1 - self.displayBmask) * self.realA 
              
        if not self.isTrain or self.cycle_consistency_loss:
            self.warpedA = self.realB
            self.warpedB = self.realA
            
        # mask(Alpha) output and BGR output
        self.maskA = self.DecoderA(self.Encoder(self.warpedA))[:,:,:,1]
        self.outputA = self.DecoderA(self.Encoder(self.warpedA))[:,:,:,1:]
        
        self.maskB = self.DecoderB(self.Encoder(self.warpedB))[:,:,:,1]
        self.outputB = self.DecoderB(self.Encoder(self.warpedB))[:,:,:,1:]
        
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
        loss_D_A = loss.adversarial_loss_discriminator(self.fakeA, self.realA, method='L2', loss_weight_config)
        loss_D_A.backward()
        
    def backward_D_B(self):
        loss_D_B = loss.adversarial_loss_discriminator(self.fakeB, self.realB, method='L2', loss_weight_config)
        loss_D_B.backward()
      
    def backward_G_A(self):
        loss_G_A = loss.adversarial_loss_generator(self.fakeA, method='L2', loss_weigth_config)
        loss_G_A += loss.reconstruction_loss(self.fakeA, self.realA, method='L2', loss_weight_config)
        loss_G_A.backward()
        
    def backward_G_B(self):
        loss_G_A = loss.adversarial_loss_generator(self.fakeA, method='L2', loss_weigth_config)
        loss_G_A += loss.reconstruction_loss(self.fakeA, self.realA, method='L2', loss_weight_config)
        loss_G_A.backward()
        
    def backward_Cycle_A(self):
        loss_Cycle_A = loss.cycle_consistency_loss(self.realA, self.cycleA, method='L2', loss_weight_config)
        loss_Cycle_A.backward()
        
    def backward_Cycle_B(self):
        loss_Cycle_B = loss.cycle_consistency_loss(self.realB, self.cycleB, method='L2', loss_weight_config)
        loss_Cycle_B.backward()
        
    def optimize_parameter(self);
    
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
    
