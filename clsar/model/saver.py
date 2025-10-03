# clsar/model/saver.py

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 10:46:59 2022

Modified to allow saving at epoch 0 without error.
"""

import torch
import os
from copy import deepcopy

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous best, then save the
    model state. Allows saving even if the best epoch is 0.
    """
    def __init__(self, 
                 data_transformer, 
                 save_dir: str = './outputs', 
                 save_name: str = 'best_model.pt', 
                 best_valid_loss: float = float('inf')):
        self.best_valid_loss = best_valid_loss
        self.save_dir = save_dir
        self.save_name = save_name
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # 先保存 data_transformer，epoch 及模型快照在 __call__ 时写入
        self.inMemorySave = {'data_transformer': data_transformer}
    
    def __call__(
        self, current_valid_loss: float, 
        epoch: int, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer
    ):
        # 当验证损失改善时，更新内存中的存档
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            smodel = deepcopy(model)
            # 直接记录传入的 epoch（可以为 0）
            self.inMemorySave.update({
                'epoch': epoch,
                'model_args': smodel.model_args,
                'model_state_dict': smodel.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            })

    def save(self):
        # 如果没有 'epoch' 字段，则默认 best_epoch = 0
        best_epoch = self.inMemorySave.get('epoch', 0)
        print("Saving final model...")
        print(f"Best validation loss: {self.best_valid_loss:.4f}")
        print(f"Saving best model at epoch: {best_epoch}\n")
        torch.save(
            self.inMemorySave,
            os.path.join(self.save_dir, self.save_name)
        )
