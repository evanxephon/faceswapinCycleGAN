import torch
from torchvision import transforms

def calc_loss(output, target, method='L2'):
    
    mse = torch.nn.MSELoss(reduce=True, size_average=True)
    
    if method == 'L2':
        loss = mse(output, target)
            
    if method == 'L1':
        loss = torch.sum(torch.abs(output - target))
        
    if method == 'CE':
        loss = mse(output, target)
    
    return loss
        
def reconstruction_loss(output, target, method='L2', loss_weight_config={}):
    
    weight = loss_weight_config['reconstruction_loss']
    
    return weight * calc_loss(output, target, method=method)

def adversarial_loss_discriminator(output_fake, output_real, method='L2', loss_weight_config={}):
    
    weight = loss_weight_config['adversarial_loss_discriminator']
    
    real = torch.ones(output_real.size())
    fake = torch.zeros(output_fake.size())    
    
    return weight * ( calc_loss(output_fake, fake, method=method) + calc_loss(output_real, real, method=method) )
    
def adversarial_loss_generator(output_fake, method='L2', loss_weight_config={}):
    
    weight = loss_weight_config['adversarial_loss_generator']
    
    fake = torch.zeros(output_fake.size())
    
    return weight * calc_loss(output_fake, fake, method=method)

def cycle_consistency_loss(input_real, output, method='L2', loss_weight_config={}):
    
    weight = loss_weight_config['cycle_consistency_loss']
    
    return weight * calc_loss(input_real, output, method=method)   

def perceptual_loss(input_real, fake, vggface, method='L2',loss_weight_config={}):

    weight = loss_weight_config['perceptual_loss']
    def preprocess_vggface(x):
        x = (x + 1)/2 * 255 # channel order: BGR
        x -= [91.4953, 103.8827, 131.0912]
        return x    
    
    real_sz224 = transform.functional.resize(input_real, [224, 224])
    real_sz224 = Lambda(preprocess_vggface)(real_sz224)
    fake_sz224 = transform.functional.resize(fake, [224, 224])
    fake_sz224 = Lambda(preprocess_vggface)(fake_sz224)
    real_feat112, real_feat55, real_feat28, real_feat7 = vggface(real_sz224)
    fake_feat112, fake_feat55, fake_feat28, fake_feat7  = vggface(fake_sz224)
    
    # Apply instance norm on VGG(ResNet) features
    # From MUNIT https://github.com/NVlabs/MUNIT
    PL = 0
    def instnorm(): return InstanceNormalization()
    
    PL += weights[0] * calc_loss(instnorm()(fake_feat7), instnorm()(real_feat7), "l2") 
    PL += weights[1] * calc_loss(instnorm()(fake_feat28), instnorm()(real_feat28), "l2")
    PL += weights[2] * calc_loss(instnorm()(fake_feat55), instnorm()(real_feat55), "l2")
    PL += weights[3] * calc_loss(instnorm()(fake_feat112), instnorm()(real_feat112), "l2")
    
    return PL
