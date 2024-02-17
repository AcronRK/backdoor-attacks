import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

import models

class TrainModel:
    def get_model_architecture(self, model_name):
        self.model = models.Models(model=model_name)    
        
    def train_epoch(self, train_loader: torch.utils.data.dataloader.DataLoader, optimizer: torch.optim.Optimizer, 
                    criterion:nn.modules.loss._Loss, device:str='cuda:0' if torch.cuda.is_available() else 'cpu'):
        # set NN into training mode
        self.model.train(True)
        
        # load model to device
        self.model.to(device)
        # extract batch size from data loader
        batch_size = train_loader.batch_size
        
        # define variables for tracking model
        running_loss = 0.0
        running_acc = 0.0
        
        # iterate over the data loader
        for batch_index, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs) 
            correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            
            running_acc += correct / batch_size
            running_loss += loss.item()

            # print every 100 batches
            if (batch_index + 1) % 100 == 0:
                running_loss_500_batches = running_loss / 100
                running_acc_500_batches = (running_acc / 100) * 100
                print(f"Iteration: {batch_index+1} - Running loss: {running_loss_500_batches:.2f} - Running acc: {running_acc_500_batches:.2f}")
            
                # reset running loss and acc
                running_loss = 0.0
                running_acc = 0.0
        print()
    
    def train_model(self, train_loader: torch.utils.data.dataloader.DataLoader, epochs: int, optimizer: str, lr: int, criterion=nn.CrossEntropyLoss(), device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            train_data (_type_): Training data
            optimizer (String): The Optimizer algorithm to use
            loss_rate (Float): loss rate
            criterion (_type_, optional): Criterion of loss. Defaults to nn.CrossEntropyLoss().
        """
        if optimizer.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            print("Optimizer not recognized")
            return
        
        # check if user correctly assigned a criterion from the loss methods
        assert(isinstance(criterion, nn.modules.loss._Loss))
    
        # start training
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            self.train_epoch(train_loader=train_loader, optimizer=self.optimizer, criterion=criterion)
            
        print("Finished!!!")
        return self.model