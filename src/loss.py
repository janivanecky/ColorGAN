import torch
import torch.nn.functional as F


def loss_gen(d_fake, d_real):
    """Vanilla GAN generator loss."""
    ones = torch.ones(d_fake.size()[0], 1)
    loss = F.binary_cross_entropy_with_logits(d_fake, ones)
    return loss

def loss_dis(d_fake, d_real):
    """Vanilla GAN discriminator loss."""
    ones = torch.ones(d_real.size()[0], 1)
    zeros = torch.zeros(d_fake.size()[0], 1)
    loss = F.binary_cross_entropy_with_logits(d_fake, zeros) + F.binary_cross_entropy_with_logits(d_real, ones)
    return loss

def loss_gen_rgan(d_fake, d_real):
    """Relativistic GAN generator loss."""
    ones = torch.ones(d_fake.size()[0], 1)
    loss = F.binary_cross_entropy_with_logits(d_fake - d_real, ones)
    return loss

def loss_dis_rgan(d_fake, d_real):
    """Relativistic GAN discriminator loss."""
    ones = torch.ones(d_real.size()[0], 1)
    loss = F.binary_cross_entropy_with_logits(d_real - d_fake, ones)
    return loss