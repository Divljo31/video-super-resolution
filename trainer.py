#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import vgg19
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import os


# In[2]:


class Trainer:
    def __init__(self, model, loss, learning_rate, checkpoint_dir='./ckpt'):
        self.model = model
        self.loss = loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, 'model_checkpoint.pth')
        self.best_psnr = -1

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.restore()

    def train(self, train_loader, valid_loader, steps, evaluate_every=1000, save_best_only=False):
        loss_mean = 0.0
        num_batches = len(train_loader)
        step = 0

        self.model.train()

        start_time = time.perf_counter()

        for epoch in range((steps // num_batches) + 1):
            for lr, hr in train_loader:
                step += 1

                lr, hr = lr.to(self.model.device), hr.to(self.model.device)
                loss = self.train_step(lr, hr)
                loss_mean += loss.item()

                if step % evaluate_every == 0:
                    avg_loss = loss_mean / evaluate_every
                    loss_mean = 0.0

                    psnr_value = self.evaluate(valid_loader)

                    duration = time.perf_counter() - start_time
                    print(f'{step}/{steps}: loss = {avg_loss:.3f}, PSNR = {psnr_value:.3f} ({duration:.2f}s)')

                    if save_best_only and psnr_value <= self.best_psnr:
                        start_time = time.perf_counter()
                        continue

                    self.best_psnr = psnr_value
                    self.save_checkpoint()

                    start_time = time.perf_counter()

                if step >= steps:
                    break

    def train_step(self, lr, hr):
        self.optimizer.zero_grad()

        sr = self.model(lr)
        loss = self.loss(sr, hr)

        loss.backward()
        self.optimizer.step()

        return loss

    def evaluate(self, data_loader):
        self.model.eval()
        total_psnr = 0.0

        with torch.no_grad():
            for lr, hr in data_loader:
                lr, hr = lr.to(self.model.device), hr.to(self.model.device)
                sr = self.model(lr)
                total_psnr += self.psnr(sr, hr)

        avg_psnr = total_psnr / len(data_loader)
        self.model.train()
        return avg_psnr

    def psnr(self, sr, hr):
        mse = nn.functional.mse_loss(sr, hr)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

    def save_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_psnr': self.best_psnr,
        }
        torch.save(checkpoint, self.checkpoint_file)
        print(f'Checkpoint saved at {self.checkpoint_file}')

    def restore(self):
        if os.path.exists(self.checkpoint_file):
            checkpoint = torch.load(self.checkpoint_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_psnr = checkpoint['best_psnr']
            print(f'Model restored from checkpoint with PSNR: {self.best_psnr:.3f}')


# In[3]:
class SrganGeneratorTrainer(Trainer):
    def __init__(self, model, checkpoint_dir, learning_rate=1e-4):
        super().__init__(model, loss=nn.MSELoss(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)



class SrganTrainer:
    def __init__(self, generator, discriminator, content_loss='VGG54', learning_rate=1e-4):
        self.generator = generator
        self.discriminator = discriminator

        # Load VGG19 model for content loss
        if content_loss == 'VGG22':
            self.vgg = self._vgg_22()
        elif content_loss == 'VGG54':
            self.vgg = self._vgg_54()
        else:
            raise ValueError("content_loss must be either 'VGG22' or 'VGG54'")

        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate)

        self.binary_cross_entropy = nn.BCELoss()
        self.mean_squared_error = nn.MSELoss()

    def train(self, train_loader, steps=200000):
        perceptual_loss_mean = 0.0
        discriminator_loss_mean = 0.0
        num_batches = len(train_loader)
        step = 0

        for epoch in range((steps // num_batches) + 1):
            for lr, hr in train_loader:
                step += 1

                lr, hr = lr.to(self.generator.device), hr.to(self.generator.device)

                perceptual_loss, discriminator_loss = self.train_step(lr, hr)
                perceptual_loss_mean += perceptual_loss.item()
                discriminator_loss_mean += discriminator_loss.item()

                if step % 50 == 0:
                    avg_perceptual_loss = perceptual_loss_mean / 50
                    avg_discriminator_loss = discriminator_loss_mean / 50
                    print(f'{step}/{steps}, perceptual loss = {avg_perceptual_loss:.4f}, discriminator loss = {avg_discriminator_loss:.4f}')
                    perceptual_loss_mean = 0.0
                    discriminator_loss_mean = 0.0

                if step >= steps:
                    break

    def train_step(self, lr, hr):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

        sr = self.generator(lr)

        hr_output = self.discriminator(hr)
        sr_output = self.discriminator(sr)

        con_loss = self._content_loss(hr, sr)
        gen_loss = self._generator_loss(sr_output)
        perc_loss = con_loss + 0.001 * gen_loss
        disc_loss = self._discriminator_loss(hr_output, sr_output)

        perc_loss.backward(retain_graph=True)
        disc_loss.backward()

        self.generator_optimizer.step()
        self.discriminator_optimizer.step()

        return perc_loss, disc_loss

    def _content_loss(self, hr, sr):
        sr = self.preprocess_input(sr)
        hr = self.preprocess_input(hr)
        sr_features = self.vgg(sr) / 12.75
        hr_features = self.vgg(hr) / 12.75
        return self.mean_squared_error(sr_features, hr_features)

    def _generator_loss(self, sr_out):
        return self.binary_cross_entropy(torch.ones_like(sr_out), sr_out)

    def _discriminator_loss(self, hr_out, sr_out):
        hr_loss = self.binary_cross_entropy(torch.ones_like(hr_out), hr_out)
        sr_loss = self.binary_cross_entropy(torch.zeros_like(sr_out), sr_out)
        return hr_loss + sr_loss

    def preprocess_input(self, x):
        x = x * 0.5 + 0.5  # De-normalize to [0, 1]
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)

    def _vgg_22(self):
        vgg = vgg19(pretrained=True).features[:5]
        return nn.Sequential(*vgg).eval()

    def _vgg_54(self):
        vgg = vgg19(pretrained=True).features[:20]
        return nn.Sequential(*vgg).eval()


# In[ ]:




