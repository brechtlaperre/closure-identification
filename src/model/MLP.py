#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 9 08:41:44 2020

@author: Brecht Laperre
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

class MLP(nn.Module):
    def __init__(self, config, input_size, output_size, input_scaler=None, output_scaler=None, seed=1234, device='cpu', relu=False):
        super(MLP, self).__init__()
        self.seed = seed
        torch.manual_seed(seed)

        if 'lr' in config.keys():
            self.learning_rate = config['lr']
        else:
            self.learning_rate = 10**config['lr_power']
        
        self.device = device
        
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        print(input_size, output_size)
        self.model = self._create_mlp_from_config(config, input_size, output_size, relu)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                y = m.in_features
                # m.weight.data shoud be taken from a normal distribution
                m.weight.data.normal_(0.0,1/np.sqrt(y))
                # m.bias.data should be 0
                m.bias.data.fill_(0)

        self.model.apply(init_weights)
        self.model = self.model.to(device)
        
    def forward(self, x):
        # Watch out! This assumes the data has already been correctly transformed
        x = x.to(self.device)
        return self.model(x)

    def predict(self, x):
        # X is a numpy matrix
        self.eval()
        self.model.eval()
        if self.input_scaler:
            x_t = self.input_scaler.transform(x)
        else:
            x_t = x
        if x_t.shape[0] > 10**5:
            input = TensorDataset(torch.Tensor(x_t).to(self.device))
            loader = DataLoader(dataset=input, batch_size=30000)
            batches = []
            for x_batch in loader:
                y_batch = self.model(x_batch[0])
                batches.append(y_batch.detach().cpu().numpy())
            
            y = np.concatenate(batches)
        else:
            x_t = torch.Tensor(x_t).to(self.device)
            y = self.model(x_t)
            y = y.detach().cpu().numpy()

        if self.output_scaler:
            y = self.output_scaler.inverse_transform(y)
        return y
    
    def fit(self, X, Y, X_val, Y_val, batch_size, num_epochs):
        X, Y, X_val, Y_val = X.copy(), Y.copy(), X_val.copy(), Y_val.copy()
        
        if self.input_scaler:
            X = self.input_scaler.fit_transform(X)
            X_val = self.input_scaler.transform(X_val)
        
        if self.output_scaler:
            Y = self.output_scaler.fit_transform(Y)
            Y_val = self.output_scaler.transform(Y_val)

        training = TensorDataset(torch.Tensor(X).to(self.device), torch.Tensor(Y).to(self.device))
        validation = TensorDataset(torch.Tensor(X_val).to(self.device), torch.Tensor(Y_val).to(self.device))

        trainloader = DataLoader(dataset=training, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
        validloader = DataLoader(dataset=validation, batch_size=2048, num_workers=0, pin_memory=False)
        print("Training the following nn:")
        print(self)
        criterion = nn.MSELoss()
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True, cooldown=15)
        
        min_val = np.inf
        best_model = self.model.state_dict()
        best_epoch = 0
        
        tr_loss = []
        val_loss = []
        print("Training ...")
        for epoch in range(num_epochs):
            self.train()
            self.model.train()
            running_loss = 0
            for data in trainloader:
                x, y = data
                
                output = self(x)
                loss = criterion(output, y)
                
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * len(y)
            else:
                loss = running_loss / len(trainloader)
                tr_loss.append(loss)
            scheduler.step(loss)

            self.eval()
            self.model.eval()
            with torch.no_grad():
                vloss = 0
                for data in validloader:
                    x, y = data
                    output = self.model(x)
                    loss = criterion(output, y)
                    vloss += loss.item() * len(y)
                else:
                    vloss = vloss / len(validloader)
                    val_loss.append(vloss)
                    if vloss < min_val:
                        best_epoch = epoch
                        min_val = vloss
                        best_model = self.model.state_dict()
                         
            print("Epoch = {:d} : Train loss = {:08.6f}; Valid loss = {:08.6f}".format(epoch+1, tr_loss[-1], val_loss[-1]))
        
        self.model.load_state_dict(best_model)
        print('Best epoch: {}  --  Validation loss: {} '.format(best_epoch, min_val))
        return tr_loss, val_loss
    
    @staticmethod
    def _create_mlp_from_config(config, inputsize, outputsize, relu=False):
        layers = []
        in_features = inputsize
        for i in range(config['n_layers']):
            out_features = config["n_units_l{}".format(i)]
            layers.append(nn.Linear(in_features, out_features))
            layer_function = config['layer_function_l{}'.format(i)]
            if layer_function == 'Dropout':
                p = config["dropout_l{}".format(i)]
                lf = nn.Dropout(p)
            else:
                lf = getattr(nn, layer_function)()
            layers.append(lf)
            #layers.append(nn.BatchNorm1d(out_features))

            in_features = out_features
        layers.append(nn.Linear(in_features, outputsize))
        if relu:
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

   



