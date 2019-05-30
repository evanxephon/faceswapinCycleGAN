import torch
import torch.nn as nn

def calc_loss(output, target, method='L2'):
    
    mse = torch.nn.MSELoss(reduce=True, size_average=True).cuda()
    
    if method == 'L2':
        loss = mse(output, target)
            
    elif method == 'L1':
        loss = torch.sum(torch.abs(output - target))
        
    elif method == 'CE':
        loss = mse(output, target)
    
    return loss
        
def reconstruction_loss(output, target, method='L2', loss_weight_config={}):
    
    weight = torch.tensor(loss_weight_config['reconstruction_loss']).cuda()
    
    return weight * calc_loss(output, target, method=method)

def adversarial_loss_discriminator(output_fake, output_real, method='L2', loss_weight_config={}):
    
    weight = torch.tensor(loss_weight_config['adversarial_loss_discriminator']).cuda()
    
    real = torch.ones(output_real.size()).cuda()
    fake = torch.zeros(output_fake.size()).cuda()    
    
    return weight * ( calc_loss(output_fake, fake, method=method) + calc_loss(output_real, real, method=method) )
    
def adversarial_loss_generator(output_fake, method='L2', loss_weight_config={}):
    
    weight = torch.tensor(loss_weight_config['adversarial_loss_generator']).cuda()
    
    fake = torch.zeros(output_fake.size()).cuda()
    
    return weight * calc_loss(output_fake, fake, method=method)

def cycle_consistency_loss(input_real, output, method='L2', loss_weight_config={}):
    
    weight = torch.tensor(loss_weight_config['cycle_consistency_loss']).cuda()
    
    return weight * calc_loss(input_real, output, method=method)   

def perceptual_loss(input_real, fake, vggface, vggface_ft_pl, method='L2',loss_weight_config={}):

    weights = torch.tensor(loss_weight_config['perceptual_loss']).cuda()
    
    def preprocess_vggface(x):
        x = (x + 1)/2 * 255 # channel order: BGR
        x -= torch.tensor([91.4953, 103.8827, 131.0912])[None,:,None,None].float().cuda()
        return x

    real = nn.functional.interpolate(input_real, (224,224))
    fake = nn.functional.interpolate(fake, (224,224))
    
    # rgb to bgr
    
    # preprocess accroding to the vggface model
    real = preprocess_vggface(real).cuda()
    fake = preprocess_vggface(fake).cuda()
    
    # vggface forward 
    with torch.no_grad():
        vggface(real)
    
    # get feature map from hook 
    real_ft_l1 = vggface_ft_pl.featuremaps[0]
    real_ft_l2 = vggface_ft_pl.featuremaps[1]
    real_ft_l3 = vggface_ft_pl.featuremaps[2]
    real_ft_l4 = vggface_ft_pl.featuremaps[3]
    
    with torch.no_grad():
        vggface(fake)
    
    fake_ft_l1 = vggface_ft_pl.featuremaps[0]
    fake_ft_l2 = vggface_ft_pl.featuremaps[1]
    fake_ft_l3 = vggface_ft_pl.featuremaps[2]
    fake_ft_l4 = vggface_ft_pl.featuremaps[3]
    
    # Apply instance norm on VGG(ResNet) features
    # From MUNIT https://github.com/NVlabs/MUNIT
    PL = 0
    
    PL += weights[0] * calc_loss(nn.functional.instance_norm(real_ft_l1), nn.functional.instance_norm(fake_ft_l1), 'L1') 
    PL += weights[1] * calc_loss(nn.functional.instance_norm(real_ft_l2), nn.functional.instance_norm(fake_ft_l2), 'L1')
    PL += weights[2] * calc_loss(nn.functional.instance_norm(real_ft_l3), nn.functional.instance_norm(fake_ft_l3), 'L1')
    PL += weights[3] * calc_loss(nn.functional.instance_norm(real_ft_l4), nn.functional.instance_norm(fake_ft_l4), 'L1')
    
    return PL
