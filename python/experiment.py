''' Copyright 2022 Andrea Rafanelli. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License '''

author = 'Andrea Rafanelli'


import torch
import copy
from utils.dataset import GetDataset
from utils.network import Trainer, Tester
import segmentation_models_pytorch as sm
import argparse
import os

class SegmentationExperiment:
    
    def __init__(self, expName, encoder='resnet152', weights='imagenet', num_epochs=250, learning_rate=5e-2, momentum=0.95):
        
        self.labels = {'Background':0,'Building':1,'Road':2, 'Water': 3,'Tree':4,'Vehicle':5,'Pool':6,'Grass':7}
        self.encoder = encoder
        self.weights = weights
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.expName = expName
        self.best_loss = 1e10
        self.best_acc = 0
        self.best_epoch = 0
        self.best_miou = 0
        

        self.dataset = GetDataset(self.labels)
        self.dataset.initialize_data()
        self.train_loader, self.test_loader, self.val_loader = self.dataset.get_loaders()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = sm.PSPNet(
            encoder_name = self.encoder, 
            encoder_weights = self.weights, 
            classes = len(self.labels),
        )
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.learning_rate, momentum = self.momentum)


    def run(self):
        torch.cuda.empty_cache()
        for epoch in range(self.num_epochs):
            print('*' * 40)
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('*' * 40)
            trainer = Trainer(self.model, self.optimizer, self.device)
            train_epoch = trainer.training(self.train_loader, 'Train')
            test_epoch = trainer.testing(self.test_loader, 'Test')
            
            if test_epoch['MIoU'] > self.best_miou:
                self.best_miou = test_epoch['MIoU']
                self.best_epoch = epoch
                self.best_model = copy.deepcopy(model.state_dict())
                print(f'Best miou: {self.best_miou:.4f} Epoch: {epoch +1}')
                print(">>>>> Saving model..")
                torch.save(self.best_model, f"{os.getcwd()}/models/{self.expName}.pt")

    def load_best_model(self):
        self.model.load_state_dict(torch.load(f"{os.getcwd()}/models/{self.expName}.pt"))
        trainer = Tester(self.model, self.optimizer, self.device)
        train_epoch = trainer.testing(self.val_loader, 'Validation')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='choose to train or test the model')
    args = parser.parse_args()
    exp = SegmentationExperiment(expName = 'experiment')
    
    if args.mode == 'train':     
        exp.run()
    elif args.mode == 'test':
        exp.load_best_model()
    else:
        print("Invalid argument. Please enter 'train' or 'test'.")
