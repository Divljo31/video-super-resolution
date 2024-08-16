#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from torchmetrics import PeakSignalNoiseRatio
import matplotlib.pyplot as plt
from PIL import Image


# In[18]:


# mean RGB values of div2k image set

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255


# In[17]:


# applying model to single/batch photos
def resolve_single(model, lr):
    lr = lr.unsqueeze(0)  # Add batch dimension
    sr = resolve(model, lr)
    return sr.squeeze(0)  # Remove batch dimension

def resolve(model, lr_batch):
    with torch.no_grad():
        lr_batch = lr_batch.float()
        sr_batch = model(lr_batch)
        sr_batch = torch.clamp(sr_batch, 0, 255)
        sr_batch = torch.round(sr_batch)
        sr_batch = sr_batch.byte()
    return sr_batch


# In[5]:


def evaluate(model, dataloader):
    psnr_values = []
    psnr_metric = PeakSignalNoiseRatio(data_range=255.0)
    for lr, hr in dataloader:
        sr = resolve(model, lr)
        psnr_value = psnr_metric(hr, sr).item()
        psnr_values.append(psnr_value)
    return np.mean(psnr_values)


# In[14]:


def normalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return (x - torch.tensor(rgb_mean)) / 127.5

def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return x * 127.5 + torch.tensor(rgb_mean)

def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0

def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1

def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5


# In[15]:


def psnr(x1, x2):
    psnr_metric = PeakSignalNoiseRatio(data_range=255.0)
    return psnr_metric(x1, x2)
def pixel_shuffle(scale):
    return torch.nn.PixelShuffle(scale)


# In[16]:


def load_image(path):
    return np.array(Image.open(path))


def plot_sample(lr, sr):
    plt.figure(figsize=(20, 10))

    images = [lr, sr]
    titles = ['LR', f'SR (x{sr.shape[0] // lr.shape[0]})']

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 2, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])

