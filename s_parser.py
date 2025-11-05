
import argparse
import torch

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
        
def parse_args():
    parser = argparse.ArgumentParser(description="SwiftWind")

    # Data
    parser.add_argument("--num_sensors", default=50, type=int)
    parser.add_argument("--gpu_device", default=0, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--batch_frames", default=64, type=int)
    parser.add_argument("--batch_pixels", default=2048, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--accum_grads", default=1, type=int)
    parser.add_argument("--noise_std", default=0.0, type=float)
    
    # Positional Encodings
    parser.add_argument("--space_bands", default=32, type=int)
    
    # Checkpoints
    parser.add_argument("--load_model_num", default=None, type=int)
    parser.add_argument("--test", default=False, type=str2bool)
    
    # Encoder
    parser.add_argument("--enc_preproc_ch", default=64, type=int)
    parser.add_argument("--num_latents", default=4, type=int)
    parser.add_argument("--enc_num_latent_channels", default=16, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--num_cross_attention_heads", default=2, type=int)
    parser.add_argument("--enc_num_self_attention_heads", default=2, type=int)
    parser.add_argument("--num_self_attention_layers_per_block", default=3, type=int)
    parser.add_argument("--dropout", default=0.00, type=float)
    
    # Decoder
    parser.add_argument("--dec_preproc_ch", default=None, type=int)
    parser.add_argument("--dec_num_latent_channels", default=16, type=int)
    parser.add_argument("--dec_num_cross_attention_heads", default=1, type=int)
    
    # Train
    parser.add_argument("--max_epochs", default=-1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)

    args = parser.parse_args()
    if torch.cuda.is_available():
        accelerator = "gpu"
        gpus = [args.gpu_device]
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        gpus = [args.gpu_device]
    else:
        accelerator = "cpu"
        gpus = None
        
    data_config = dict(data_name = args.data_name,
                       num_sensors = args.num_sensors,
                       gpu_device=None if accelerator == 'cpu' else gpus,
                       accelerator = accelerator,
                       seed = args.seed,
                       batch_frames = args.batch_frames,
                       batch_pixels = args.batch_pixels,
                       lr=args.lr,
                       accum_grads = args.accum_grads,
                       test = args.test,
                       space_bands=args.space_bands,
                       noise_std=args.noise_std,
                       )
    
    encoder_config = dict(load_model_num=args.load_model_num,
                          enc_preproc_ch=args.enc_preproc_ch, 
                          num_latents=args.num_latents,     
                          enc_num_latent_channels=args.enc_num_latent_channels,  
                          num_layers=args.num_layers,
                          num_cross_attention_heads=args.num_cross_attention_heads,
                          enc_num_self_attention_heads=args.enc_num_self_attention_heads,
                          num_self_attention_layers_per_block=args.num_self_attention_layers_per_block,
                          dropout=args.dropout,
                          )

    decoder_config = dict(dec_preproc_ch=args.dec_preproc_ch,  
                          dec_num_latent_channels=args.dec_num_latent_channels, 
                          latent_size=1,  
                          dec_num_cross_attention_heads=args.dec_num_cross_attention_heads
                          )
    
    train_config = dict(max_epochs=args.max_epochs,
                        num_workers=args.num_workers)
    
    return data_config, encoder_config, decoder_config, train_config 
