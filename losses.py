import torch
import torch.nn.functional as F
import math

def g_nonsaturating(fake_scores):
    return F.softplus(-fake_scores).mean()

def d_logistic(real_scores, fake_scores):
    return F.softplus(fake_scores).mean() + F.softplus(-real_scores).mean()

def d_r1regularization(real_images, real_scores, r1_gamma=10):
    grad = torch.autograd.grad(outputs=real_scores.sum(), inputs=real_images, create_graph=True)[0]
    grad = grad.view(grad.shape[0], -1).pow(2).sum(1).mean()
    r1_penalty = (0.5 * r1_gamma) * grad
    return r1_penalty

def g_path_length_regularization(fake_images, w, ema_path_length, decay=0.99):
    noise_images = torch.randn(fake_images.shape, device=fake_images.device)
    noise_images = noise_images / math.sqrt(fake_images.shape[2] * fake_images.shape[3])
    grad = torch.autograd.grad(outputs=(fake_images * noise_images).sum(), inputs=w, create_graph=True)[0]
    
    lengths = torch.sqrt(grad.pow(2).sum(1).mean())

    ema_path_length = decay * ema_path_length + (1 - decay) * lengths
    
    path_length_penalty = lengths - ema_path_length
    path_length_penalty = path_length_penalty.pow(2)

    return path_length_penalty, ema_path_length.detach()

def update_ema(model, ema, decay=0.999):
    temp1 = dict(ema.named_parameters())
    temp2 = dict(model.named_parameters())

    for k in temp1.keys():
        temp1[k].data = decay * temp1[k].data + (1 - decay) * temp2[k].data
        
