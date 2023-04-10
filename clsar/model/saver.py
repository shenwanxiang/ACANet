# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 10:46:59 2022

@author: wanxiang.shen@u.nus.edu
"""

import torch
import os
from copy import deepcopy

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, 
                 data_transformer, 
                 save_dir = './outputs', 
                 save_name = 'best_model.pt', 
                 best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        self.save_dir = save_dir
        self.save_name = save_name
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.inMemorySave = {'data_transformer': data_transformer}
        
        
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            #print(f"\nBest validation loss: {self.best_valid_loss}")
            #print(f"\nSaving best model for epoch: {epoch+1}\n")
            smodel = deepcopy(model) #should be deepcopy
            self.inMemorySave.update({'epoch': epoch+1,
                                     'model_args': smodel.model_args,
                                     'model_state_dict':smodel.state_dict(),
                                     'optimizer_state_dict':optimizer.state_dict()
                                     })

    def save(self):
        print("Saving final model...")
        print(f"\nBest validation loss: {self.best_valid_loss}")
        print(f"\nSaving best model on epoch: {self.inMemorySave['epoch']}\n")
        torch.save(self.inMemorySave, os.path.join(self.save_dir,
                                                   self.save_name))
