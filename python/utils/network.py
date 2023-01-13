
'''
Copyright 2023 Andrea Rafanelli.
Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on 
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and limitations under the License
'''

__author__ = 'Andrea Rafanelli'



import torch
import cv2
import os
import numpy as np 
import torch.nn.functional as F
import sklearn
from collections import defaultdict
from tqdm.auto import tqdm
from sklearn.metrics import jaccard_score, accuracy_score

class Trainer:
    
    def __init__(self, model, optimizer, device):
        
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.metrics = defaultdict(float)
        self.metrics_calculator = MetricsCalculator()
        self.loss_function = LossCalculator()
    
    def training(self, dataloaders, set_name):
        self.metrics.clear()
        epoch_samples = 0
        self.model.train()
        

        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloaders)):
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels, self.metrics)
                loss.backward()
                self.optimizer.step()
                
            epoch_samples += inputs.size(0)
                
        self.metrics_calculator._update_metrics(self.metrics, epoch_samples, set_name)
        
        return self.metrics
    
        
    def testing(self, dataloaders, set_name):
        self.metrics.clear()
        epoch_samples = 0
        self.model.eval()

        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloaders)):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, labels, self.metrics)
            epoch_samples += inputs.size(0)
            
        self.metrics_calculator._update_metrics(self.metrics, epoch_samples, set_name)
        
        return self.metrics

class Tester:
    
    """Initialize the tester with the model, optimizer, device,
     batch, valid_path, metrics, metrics_calculator, and loss_function"""
    def __init__(self, model, optimizer, device):
        
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.batch = 8
        self.metrics_calculator = MetricsCalculator()
        self.loss_function = LossCalculator()
        self.metrics = defaultdict(float)
        
    def reverse_transform_mask(self, input_):
        """Reverse the transformation on the mask"""
        input_ = input_.transpose((1, 2, 0))
        t_mask = np.argmax(input_,axis=2).astype('float32') 
        t_mask = cv2.resize(t_mask, dsize=(384, 384))
        
        return t_mask

    def testing(self, dataloaders, set_name):
        """Test the model on the provided data loader,
        save the output masks, and update the metrics dictionary"""
        self.metrics.clear()
        self.model.eval()
        epoch_samples = 0

        for index, inputs, labels in tqdm(dataloaders):
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(inputs)
                preds = torch.sigmoid(outputs)
                preds = preds.data.cpu().numpy()
                loss = self.loss_function(outputs, labels, self.metrics)
                epoch_samples += inputs.size(0)
                
                for i in range(self.batch):   
                    try:
                        idx = index[i].split('/')[-1]
                        f_mask = self.reverse_transform_mask(preds[i])
                        cv2.imwrite(os.path.join(f"{os.getcwd()}/predictions/{idx}" , f_mask))
                    except:
                        continue
        self.metrics_calculator._update_metrics(self.metrics, epoch_samples, set_name)
        
        return self.metrics
    

class MetricsCalculator:
    def _update_metrics(self, metrics, epoch_samples, set_name):
        for metric in ['Bce', 'Dice', 'Loss', 'MIoU', 'Accuracy']:
            metrics[metric] /= epoch_samples
            
        self.print_metrics(metrics, set_name)

    def print_metrics(self, metrics, phase):
        outputs = []
        for k in metrics.keys():
            outputs.append("{}: {:4f}".format(k, metrics[k]))
        print("{}: {}".format(phase, ", ".join(outputs)))


class LossCalculator:

    def __init__(self, bce_weight = 0.35):

        self.bce_weight = bce_weight
        
    @staticmethod
    def diceLoss(pred, target, smooth = 1e-4):
        pred = pred.contiguous()
        target = target.contiguous()    
        intersection = (pred * target).sum(dim = 2).sum(dim = 2)
        loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim = 2) + target.sum(dim = 2).sum(dim = 2) + smooth)))
    
        return loss.mean()

    def __call__(self, pred, target, metrics):
        
        bce = F.binary_cross_entropy_with_logits(pred, target.to(torch.float32))
        pred = torch.sigmoid(pred)
        dice = self.diceLoss(pred, target)

        target_np = target.data.cpu().numpy()
        pred_np= pred.data.cpu().numpy()

        MIoU = np.mean(jaccard_score(np.argmax(target_np, axis = 1).flatten(), np.argmax(pred_np, axis = 1).flatten(), average = None))
        accuracy = accuracy_score(np.argmax(target_np, axis = 1).flatten(), np.argmax(pred_np, axis = 1).flatten())
        loss = bce * self.bce_weight + dice * (1 - self.bce_weight)
        self._update_metrics(metrics, target.size(0), bce, dice, loss, accuracy, MIoU)
        return loss
    
    def _update_metrics(self, metrics, batch_size, bce, dice, loss, accuracy, MIoU):
        metrics['Bce'] += bce.data.cpu().numpy() * batch_size
        metrics['Dice'] += dice.data.cpu().numpy() * batch_size
        metrics['Loss'] += loss.data.cpu().numpy() * batch_size
        metrics['Accuracy'] += accuracy * batch_size
        metrics['MIoU'] += MIoU * batch_size
