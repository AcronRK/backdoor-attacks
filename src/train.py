import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import pandas as pd

import models

class TrainModel:
    def __init__(self, model_name, device:str='cuda:0' if torch.cuda.is_available() else 'cpu'):
        self.model = models.Models(model=model_name)  
        # load model to device
        self.model.to(device)  
        # init best_val_loss variable
        self.best_val_loss = float('inf')
        self.best_model_state = None

        
    def train_epoch(self, train_loader: torch.utils.data.dataloader.DataLoader, optimizer: torch.optim.Optimizer, 
                    criterion:nn.modules.loss._Loss, device:str='cuda:0' if torch.cuda.is_available() else 'cpu'):
        # set NN into training mode
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            self.scheduler.step()
            predicted = torch.argmax(outputs, dim=1)  # Get the predicted class labels
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples
        
        return epoch_loss, epoch_accuracy
        
    def validate(self, val_loader, criterion, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        self.model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = correct_predictions / total_samples
        
        return avg_val_loss, avg_val_accuracy
    
    
    def train_model(self, train_loader: torch.utils.data.dataloader.DataLoader, val_loader: torch.utils.data.dataloader.DataLoader,
                    epochs: int, optimizer: str, lr: int, criterion=nn.CrossEntropyLoss(), device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            train_data (_type_): Training data
            optimizer (String): The Optimizer algorithm to use
            loss_rate (Float): loss rate
            criterion (_type_, optional): Criterion of loss. Defaults to nn.CrossEntropyLoss().
        """
        if optimizer.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        else:
            print("Optimizer not recognized")
            return
        
        
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 30, 50], gamma=0.5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        # check if user correctly assigned a criterion from the loss methods
        assert(isinstance(criterion, nn.modules.loss._Loss))
    
    
        # init local val loss and model state
        best_val_loss = float('inf')
        best_model = None
        
        # variables to save training and validation losses and accuracies
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train_loss, train_accuracy = self.train_epoch(train_loader=train_loader, optimizer=self.optimizer, criterion=criterion)
            print(f"Training Loss: {train_loss:.4f} | Accuracy: {train_accuracy:.4f}")
            
            val_loss, val_accuracy = self.validate(val_loader, criterion)
            print(f"Validation Loss: {val_loss:.4f} | Accuracy: {val_accuracy:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model
                
            # save the values
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
        # create dataframe with values and save as csv
        df = pd.DataFrame({
            'train_loss': train_losses,
            'train_accuracy': train_accuracies,
            'val_loss': val_losses,
            'val_accuracy': val_accuracies
        })
                
        print("Finished Training")
        return best_model, df
    
    
    
    