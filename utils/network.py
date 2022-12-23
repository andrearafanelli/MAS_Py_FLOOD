
'''
Copyright 2022 Andrea Rafanelli.
Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on 
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and limitations under the License
'''

__author__ = 'Andrea Rafanelli'



import cv2
import torch
import numpy as np 
import torch.nn.functional as F
import sklearn
from collections import defaultdict
from tqdm.auto import tqdm

class Dataset:
    
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
            return index, image, mask
        else:
            return image, mask
        
        
        
def training(model, lossFunction, dataloaders, optimizer, device, setName):
    
    model.train()
    metrics = defaultdict(float)
    epoch_samples = 0
    
    for batch_idx, (inputs, labels) in enumerate(tqdm(dataloaders)):
        
        inputs = inputs.to(device)
        labels = labels.to(device)
               
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            
            outputs = model(inputs)
            loss = lossFunction(outputs, labels, metrics)
            loss.backward()
            optimizer.step()
        
        epoch_samples += inputs.size(0)
                                                 
    Metrics(metrics, epoch_samples, setName)
    epoch_loss = metrics['Loss'] / epoch_samples
    epoch_miou = metrics['MIoU'] / epoch_samples
    epoch_acc = metrics['Accuracy'] / epoch_samples
                                           
    return epoch_loss, epoch_miou, epoch_acc


def validating(model, lossFunction, dataloaders, optimizer, device, setName):
    
    model.eval()
    metrics = defaultdict(float)
    epoch_samples = 0
    
    for batch_idx, (inputs, labels) in enumerate(tqdm(dataloaders)):
        
        inputs = inputs.to(device)
        labels = labels.to(device)
                    
        optimizer.zero_grad()
        outputs = model(inputs)
       
        loss = lossFunction(outputs, labels, metrics)
        
        epoch_samples += inputs.size(0)
                                                 
    Metrics(metrics, epoch_samples, setName)
    epoch_loss = metrics['Loss'] / epoch_samples
    epoch_miou = metrics['MIoU'] / epoch_samples
    epoch_acc = metrics['Accuracy'] / epoch_samples
                                                 
    return epoch_loss, epoch_miou, epoch_samples


def diceLoss(pred, target, smooth = 1e-5):
    
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim = 2).sum(dim = 2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim = 2) + target.sum(dim = 2).sum(dim = 2) + smooth)))
    
    return loss.mean()


def lossCalculation(pred, target, metrics, bce_weight = 0.5):

    bce = F.binary_cross_entropy_with_logits(pred, target.to(torch.float32))
    pred = torch.sigmoid(pred)
    dice = diceLoss(pred, target)
    accuracy = torch.sum(pred == target)

    target_np = target.data.cpu().numpy()
    pred_np= pred.data.cpu().numpy()
    MIoU = np.mean(sklearn.metrics.jaccard_score(np.argmax(target_np, axis = 1).flatten(), np.argmax(pred_np, axis = 1).flatten(), average = None))
    accuracy = sklearn.metrics.accuracy_score(np.argmax(target_np, axis = 1).flatten(), np.argmax(pred_np, axis = 1).flatten())
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    metrics['Bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['Dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['Loss'] += loss.data.cpu().numpy() * target.size(0)
    metrics['Accuracy'] += accuracy * target.size(0)
    metrics['MIoU'] += MIoU * target.size(0)

    return loss

def Metrics(metrics, epoch_samples, phase):
    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format(phase, ", ".join(outputs)))
