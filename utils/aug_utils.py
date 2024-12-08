import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import PIL

pil_to_tensor = transforms.ToTensor()
tensor_to_pil = transforms.ToPILImage()

def hausdorff_distance(tensor1, tensor2):
    distmat = torch.cdist(tensor1, tensor2)
    hd1 = torch.max(torch.min(distmat, dim=1)[0])
    hd2 = torch.max(torch.min(distmat, dim=0)[0])
    return max(hd1, hd2)

def compute_snr(tensor1, tensor2):
    signal = torch.norm(tensor1)
    noise = torch.norm(tensor1 - tensor2)
    snr = 20 * torch.log10(signal / noise)
    return snr

class addNoise(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, sigma):
        super(addNoise, self).__init__()
        self.sigma = sigma


    def forward(self, noised_and_cover):
        
        if isinstance(noised_and_cover, PIL.Image.Image):
            noised_and_cover = pil_to_tensor(noised_and_cover)

        noised_image = noised_and_cover
        noised_and_cover = noised_image + (self.sigma ** 2) * torch.randn_like(noised_image)

        noised_and_cover = torch.clamp(noised_and_cover, 0, 1)
        noised_and_cover = tensor_to_pil(noised_and_cover)

        return noised_and_cover


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min

class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, resize_ratio_range, interpolation_method='nearest'):
        super(Resize, self).__init__()
        self.resize_ratio_min = resize_ratio_range[0]
        self.resize_ratio_max = resize_ratio_range[1]
        self.interpolation_method = interpolation_method


    def forward(self, noised_and_cover):

        resize_ratio = random_float(self.resize_ratio_min, self.resize_ratio_max)
        noised_image = noised_and_cover
        noised_and_cover = F.interpolate(
                                    noised_image,
                                    scale_factor=(resize_ratio, resize_ratio),
                                    mode=self.interpolation_method)

        return noised_and_cover
