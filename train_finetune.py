
# train.py for arbitrary-location queries

import numpy as np
from glob import glob as gb
import torch
import time
import random
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy

from s_parser import parse_args
from dataloaders_finetune import SwiftWind_train_val_dataloaders
from network import SwiftWind

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # loading input data
    start_time = time.time() 
    data_config, encoder_config, decoder_config, train_config = parse_args()
    set_seed(data_config['seed']) 
    train_loader, val_loader = SwiftWind_train_val_dataloaders(data_config, num_workers=train_config['num_workers'])
    data_loading_time = time.time() - start_time
    print(f"Loading input data time: {data_loading_time:.2f} s.")

    # initial model
    start_time = time.time()
    model = SwiftWind(**encoder_config, **decoder_config, **data_config, **train_config)
    model_initialization_time = time.time() - start_time
    print(f"Initial model time: {model_initialization_time:.2f} s.")

    # loading trained model parameters
    if encoder_config['load_model_num']:
        model_num = encoder_config['load_model_num']
        print(f'Loading {model_num} ...')
        model_loc = gb(f"lightning_logs/version_{model_num}/checkpoints/*.ckpt")[0]
        print(f'model_loc:{model_loc}')
        state_dict = torch.load(model_loc)["state_dict"]
        model.load_state_dict(state_dict, strict=True)
    else:
        model_loc = None

    # start training
    if not data_config['test']:
        start_time = time.time()
        cbs = [ModelCheckpoint(monitor="val_loss", filename="val-{epoch:02d}", every_n_epochs=1),
               EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=200, mode="min", verbose=True)]
        trainer = Trainer(max_epochs=train_config['max_epochs'], 
                          callbacks=cbs,
                          accumulate_grad_batches=data_config['accum_grads'],
                          log_every_n_steps=data_config['num_batches'],
                          accelerator='gpu',
                          devices=-1,
                          strategy=DDPStrategy(find_unused_parameters=False),
                          precision=32)
        trainer.fit(model, train_loader, val_loader)
        model_train_time = time.time() - start_time
        print(f"Training model time: {model_train_time:.2f} s.")

if __name__=='__main__':
    main()