#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import requests
from zipfile import ZipFile
import pickle
from tqdm import tqdm


# In[5]:


class DIV2K(data.Dataset):
    def __init__(self,
                 scale=2,
                 subset='train',
                 downgrade='bicubic',
                 images_dir='.div2k/images',
                 caches_dir='.div2k/caches'):

        self._ntire_2018 = True

        _scales = [2, 3, 4, 8]

        if scale in _scales:
            self.scale = scale
        else:
            raise ValueError(f'scale must be in {_scales}')

        if subset == 'train':
            self.image_ids = range(1, 801)
        elif subset == 'valid':
            self.image_ids = range(801, 901)
        else:
            raise ValueError("subset must be 'train' or 'valid'")

        _downgrades_a = ['bicubic', 'unknown']
        _downgrades_b = ['mild', 'difficult']

        if scale == 8 and downgrade != 'bicubic':
            raise ValueError(f'scale 8 only allowed for bicubic downgrade')

        if downgrade in _downgrades_b and scale != 4:
            raise ValueError(f'{downgrade} downgrade requires scale 4')

        if downgrade == 'bicubic' and scale == 8:
            self.downgrade = 'x8'
        elif downgrade in _downgrades_b:
            self.downgrade = downgrade
        else:
            self.downgrade = downgrade
            self._ntire_2018 = False

        self.subset = subset
        self.images_dir = images_dir
        self.caches_dir = caches_dir

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(caches_dir, exist_ok=True)

        self.hr_images = self._hr_image_files()
        self.lr_images = self._lr_image_files()

        # Cache the datasets
        self._cache_dataset(self.hr_images, self._hr_cache_file())
        self._cache_dataset(self.lr_images, self._lr_cache_file())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        hr_image = Image.open(self.hr_images[idx])
        lr_image = Image.open(self.lr_images[idx])

        hr_image, lr_image = self.random_transform(lr_image, hr_image)

        return transforms.ToTensor()(lr_image), transforms.ToTensor()(hr_image)

    def random_transform(self, lr_image, hr_image):
        lr_image, hr_image = random_crop(lr_image, hr_image, scale=self.scale)
        if torch.rand(1).item() > 0.5:
            lr_image, hr_image = random_flip(lr_image, hr_image)
        lr_image, hr_image = random_rotate(lr_image, hr_image)
        return lr_image, hr_image

    def _hr_image_files(self):
        images_dir = self._hr_images_dir()
        return [os.path.join(images_dir, f'{image_id:04}.png') for image_id in self.image_ids]

    def _lr_image_files(self):
        images_dir = self._lr_images_dir()
        return [os.path.join(images_dir, self._lr_image_file(image_id)) for image_id in self.image_ids]

    def _lr_image_file(self, image_id):
        if not self._ntire_2018 or self.scale == 8:
            return f'{image_id:04}x{self.scale}.png'
        else:
            return f'{image_id:04}x{self.scale}{self.downgrade[0]}.png'

    def _hr_images_dir(self):
        return os.path.join(self.images_dir, f'DIV2K_{self.subset}_HR')

    def _lr_images_dir(self):
        if self._ntire_2018:
            return os.path.join(self.images_dir, f'DIV2K_{self.subset}_LR_{self.downgrade}')
        else:
            return os.path.join(self.images_dir, f'DIV2K_{self.subset}_LR_{self.downgrade}', f'X{self.scale}')

    def _hr_cache_file(self):
        return os.path.join(self.caches_dir, f'DIV2K_{self.subset}_HR.cache')

    def _lr_cache_file(self):
        return os.path.join(self.caches_dir, f'DIV2K_{self.subset}_LR_{self.downgrade}_X{self.scale}.cache')

    def _cache_dataset(self, image_files, cache_file):
        if not os.path.exists(cache_file):
            print(f'Caching dataset to {cache_file} ...')
            for image_file in image_files:
                # Simulate caching by accessing all images once
                Image.open(image_file).load()
            print(f'Dataset cached to {cache_file}.')

    @staticmethod
    def download_archive(file, target_dir, extract=True):
        source_url = f'http://data.vision.ee.ethz.ch/cvl/DIV2K/{file}'
        target_path = os.path.join(target_dir, file)

        # Send a request to get the file size for progress tracking
        response = requests.head(source_url)
        file_size = int(response.headers.get('content-length', 0))

        # Download the file with a progress bar
        response = requests.get(source_url, stream=True)
        with open(target_path, 'wb') as f, tqdm(
            desc=f'Downloading {file}',
            total=file_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                bar.update(len(data))

        # Extract the file
        if extract:
            with ZipFile(target_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)

        # Remove the zip file
        os.remove(target_path)
        
# -----------------------------------------------------------
#  Transformations
# -----------------------------------------------------------

def random_crop(lr_img, hr_img, hr_crop_size=96, scale=2):
    lr_crop_size = hr_crop_size // scale

    lr_w = torch.randint(0, lr_img.width - lr_crop_size + 1, (1,)).item()
    lr_h = torch.randint(0, lr_img.height - lr_crop_size + 1, (1,)).item()

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img.crop((lr_w, lr_h, lr_w + lr_crop_size, lr_h + lr_crop_size))
    hr_img_cropped = hr_img.crop((hr_w, hr_h, hr_w + hr_crop_size, hr_h + hr_crop_size))

    return lr_img_cropped, hr_img_cropped

def random_flip(lr_img, hr_img):
    if torch.rand(1).item() > 0.5:
        return lr_img.transpose(Image.FLIP_LEFT_RIGHT), hr_img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        return lr_img, hr_img

def random_rotate(lr_img, hr_img):
    angle = torch.randint(0, 4, (1,)).item() * 90
    return lr_img.rotate(angle), hr_img.rotate(angle)


# In[15]:





# In[17]:





# In[16]:





# In[ ]:




