
# network.py for Observing system simulation experiments (OSSEs)

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from model import Encoder, Decoder
import numpy as np

class SwiftWind(pl.LightningModule):
    def __init__(self,**kwargs):
    
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)
        
        pos_encoder_ch = self.hparams.space_bands*len(self.hparams.image_size)*2
        self.encoder = Encoder(
            input_ch = self.hparams.im_ch+pos_encoder_ch, 
            preproc_ch = self.hparams.enc_preproc_ch,  
            num_latents = self.hparams.num_latents,  
            num_latent_channels = self.hparams.enc_num_latent_channels, 
            num_layers = self.hparams.num_layers, 
            num_cross_attention_heads = self.hparams.num_cross_attention_heads, 
            num_self_attention_heads = self.hparams.enc_num_self_attention_heads, 
            num_self_attention_layers_per_block = self.hparams.num_self_attention_layers_per_block, 
            dropout = self.hparams.dropout,
        )
        
        self.decoder_1 = Decoder(
            ff_channels = pos_encoder_ch+1, 
            preproc_ch = self.hparams.dec_preproc_ch,  
            num_latent_channels = self.hparams.dec_num_latent_channels,  
            latent_size = self.hparams.latent_size,  
            num_output_channels = self.hparams.im_ch, 
            num_cross_attention_heads = self.hparams.dec_num_cross_attention_heads, 
            dropout = self.hparams.dropout, 
        )

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.num_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'\nThe model has {self.num_params} params \n')
        
    def forward(self, sensor_values, sensor_mask, query_coords):
        out = self.encoder(sensor_values, sensor_mask)
        delta = self.decoder_1(out, query_coords)
        return delta

    def training_step(self,batch, batch_idx):
        sensor_values, sensor_mask, coords, field_values = batch
        bg_values = coords[..., :1]
        delta_pred = self(sensor_values, sensor_mask, coords)
        pred_values = bg_values + delta_pred

        loss = F.mse_loss(pred_values, field_values, reduction='mean')
        self.log_dict({"train_loss": loss}, on_step=True, on_epoch=True,prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sensor_values, sensor_mask, coords, field_values = batch
        bg_values = coords[..., :1]
        delta_pred = self(sensor_values, sensor_mask, coords)
        pred_values = bg_values + delta_pred
        data_loss = F.mse_loss(pred_values, field_values, reduction='mean')
        self.log("val_loss", data_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return data_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)




             
    
    
    
    
    
    
    
    
    

