import torch

def calc_loss(output, target, method='L2'):
    
    if method == 'L2':
        loss = torch.sum(torch.exp((output - target),2))
            
    if method == 'L1':
        loss = torch.sum(torch.abs(output - target))
        
    if method == 'CE':
        loss = torch.sum(target * torch.log(output, 2))
        
def reconstruction_loss(output, target, method='L2', **loss_weight_config):
    
    weight = loss_weight_config['reconstruction_loss']
    
    return weight * calc_loss(output, target, method=method)

def adversarial_loss_discriminator(output_fake, output_real, method='L2', **loss_weight_config):
    
    weight = loss_weight_config['adversarial_loss']
    
    real = torch.ones(output_real.size())
    fake = torch.zeros(output_fake.size())    
    
    return weight * ( calc_loss(output_fake, fake, method=method) + calc_loss(output_real, real, method=method) )
    
def adversarial_loss_generator(output_fake, method='L2', **loss_weight_config):
    
    weight = loss_weight_config['adversarial_loss_generator']
    
    fake = torch.zeros(output_fake.size())
    
    return weight * calc_loss(output_fake, fake, method=method)

def cycle_consistency_loss(input_real, output, method='L2', **loss_weight_config):
    
    weight = loss_weight_config['cycle_consistency_loss']
    
    return weight * calc_loss(input_real, output, method=method)   
