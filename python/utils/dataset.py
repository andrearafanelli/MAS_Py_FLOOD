''' Copyright 2023 Andrea Rafanelli. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License '''

author = 'Andrea Rafanelli'

import distutils.cmd
import os
import albumentations
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader


class Dataset:
        
    """
    'x_paths': a list of paths to the images
    'label': a dictionary of labels (optional)
    'validation': a boolean that indicates whether the dataset is   for validation or not (optional)
    'augmentation': an augmentation pipeline (optional)
    'preprocessing': a preprocessing pipeline (optional)
    """

    def __init__(self, x_paths, label = None,  validation = None, augmentation = None, preprocessing = None):
        self.x_paths = x_paths
        self.y_paths = [x.replace("img", "label").replace(".jpg", "_lab.png") for x in self.x_paths]
        self.label = label 
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.validation = validation
        
    def __len__(self):
        
        return len(self.x_paths)
    
    def __getitem__(self, index):
        
        image = cv2.imread(self.x_paths[index])
        mask = cv2.imread(self.y_paths[index])
        
        if self.augmentation:
            
            sample = self.augmentation(image = image, mask = mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        mask = torch.as_tensor(mask[:,:,0], dtype=torch.int64)
        mask = torch.moveaxis(torch.nn.functional.one_hot(mask, num_classes= len(self.label)), -1, 0)
    
        image = torch.from_numpy((np.moveaxis(image,-1, 0)))
        
        if self.validation:
            return self.y_paths[index], image, mask
        else:
            return image, mask
        
        

class GetDataset(distutils.cmd.Command):

    """
    A custom command to pre-process the datasets.
    It loads the image paths from train, test, and validation folders, 
    applies the necessary image transformations, 
    creates the datasets and dataloaders for training, testing and validation.
    """

    def __init__(self, labels):
        self.train_path = '../dataset/train/img/'
        self.test_path = '../dataset/test/img/'
        self.val_path = '../dataset/val/img/'
        self.labels = labels
        self.batch_size = 8

    def _load_data(self):
    
        self.x_train = [os.path.join(self.train_path, file) for file in sorted(os.listdir(self.train_path))]
        self.x_test = [os.path.join(self.test_path, file) for file in sorted(os.listdir(self.test_path))]
        self.x_val = [os.path.join(self.val_path, file) for file in sorted(os.listdir(self.val_path))]
        
    def _create_transforms(self):
        self.train_transformation = albumentations.Compose([
            albumentations.Resize(384, 384),
            albumentations.OneOf([
                albumentations.CLAHE(p = 1),
                albumentations.HueSaturationValue(p = 1)
            ], p = 0.9),
            albumentations.IAAAdditiveGaussianNoise(p = 0.2)
        ])

        self.preprocessing = albumentations.Compose([
            albumentations.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])

        self.test_transformation = albumentations.Compose([
            albumentations.Resize(384, 384)
        ])

    def _create_datasets(self):
        self.train_dataset = Dataset(self.x_train, augmentation=self.train_transformation, preprocessing=self.preprocessing, label=self.labels)
        self.test_dataset = Dataset(self.x_test, augmentation=self.test_transformation, preprocessing=self.preprocessing, label=self.labels)
        self.val_dataset = Dataset(self.x_val, validation=True, augmentation=self.test_transformation, preprocessing=self.preprocessing, label=self.labels)

    def _create_dataloaders(self):
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def initialize_data(self):
        self._load_data()
        self._create_transforms()
        self._create_datasets()
        self._create_dataloaders()

    def get_loaders(self):
        return self.train_loader, self.test_loader, self.val_loader
    
    


