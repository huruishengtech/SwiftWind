
# test.py for Observing system simulation experiments (OSSEs)

import numpy as np
from glob import glob as gb
import torch
from s_parser import parse_args
from network import SwiftWind
from tqdm import tqdm
import os
import pandas as pd
import math
import random

# get test timesteps
def get_test_time_list_2023():
    start = np.datetime64('2023-01-01T00:00:00')
    end = np.datetime64('2023-12-31T23:00:00')
    step = np.timedelta64(60, 'm')
    time_series = np.arange(start, end + step, step, dtype='datetime64[ns]')
    time_str_list = pd.to_datetime(time_series).strftime('%Y%m%dT%H%M%S')
    return time_str_list

# lon-lat positional encoding
def latlon_fourier_encoding(latlon_tensor, num_bands=32):
    if isinstance(latlon_tensor, np.ndarray):
        latlon_tensor = torch.tensor(latlon_tensor, dtype=torch.float32)
    if latlon_tensor.shape[0] == 0:
        return torch.empty((0, 4 * num_bands), dtype=torch.float32, device=latlon_tensor.device)
    else:
        latlon_rad = latlon_tensor * math.pi / 180.0
        latlon_norm = (latlon_rad - latlon_rad.mean(dim=0)) / latlon_rad.std(dim=0)
        freqs = torch.linspace(1.0, 2 ** num_bands, num_bands, device=latlon_tensor.device) 
        lat = latlon_norm[:, 0:1]  
        lon = latlon_norm[:, 1:2] 
        lat_enc = torch.cat([torch.sin(freqs * lat), torch.cos(freqs * lat)], dim=1) 
        lon_enc = torch.cat([torch.sin(freqs * lon), torch.cos(freqs * lon)], dim=1)  
        return torch.cat([lat_enc, lon_enc], dim=1)

# define the lon-lat of GFS-6h and ERA5
def get_gfs_era5_lonlat():
    lons = np.arange(90, 150.01, 0.25)   
    lats = np.arange(40, -10.01, -0.25)   
    lon_grid, lat_grid = np.meshgrid(lons, lats)  
    latlon_flat = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=1)  
    latlon_tensor = torch.tensor(latlon_flat, dtype=torch.float32)  
    all_pos_encoding = latlon_fourier_encoding(latlon_tensor, num_bands=32)
    return all_pos_encoding

# get the coordinate vector of randomly sampled from ERA5
def wind_n_sensors(data, n_sensors, rnd_seed):
    np.random.seed(rnd_seed)
    im = np.copy(data).squeeze()
    coords = []
    for n in range(n_sensors):
        while True:
            new_x = np.random.randint(0,data.shape[0],1)[0]
            if im[new_x] != 0: 
                coords.append(new_x)
                im[new_x] = 0
                break
    coords = np.array(coords)  
    return coords

# get ERA5 data
def get_era5_data(time_sel):
    era5_dir = './final_data_used/used_model_data/era5_1h_npy_2017_2023'
    era5_name = f'era5_{time_sel}.npy'
    era5_path = os.path.join(era5_dir, era5_name)
    era5_data = np.load(era5_path)
    era5_data = np.nan_to_num(era5_data)
    return torch.tensor(era5_data.reshape(-1, 1), dtype=torch.float32)

# get GFS-6h data
def get_gfs_data(time_sel):
    gfs_dir = './final_data_used/used_model_data/gfs_1h_npy_2017_2023'
    gfs_name = f'gfs_{time_sel}.npy'
    gfs_path = os.path.join(gfs_dir, gfs_name)
    gfs_data = np.load(gfs_path)
    gfs_data = np.nan_to_num(gfs_data)
    return torch.tensor(gfs_data.reshape(-1, 1), dtype=torch.float32)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    data_config, encoder_config, decoder_config, train_config = parse_args()
    set_seed(data_config['seed']) 
    data_config['image_size'] = [201, 241]
    data_config['im_ch'] = 1
    model = SwiftWind(**encoder_config, **decoder_config, **data_config, **train_config)
    model_num = encoder_config['load_model_num']
    print(f'Loading {model_num} ...')
    model_loc = gb(f"lightning_logs/version_{model_num}/checkpoints/*.ckpt")[0]
    model = model.load_from_checkpoint(model_loc, **encoder_config, **decoder_config, **data_config, **train_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    save_path = model_loc.split('checkpoints')[0]
    era5_gfs_pos_encoding = get_gfs_era5_lonlat().to(device) 
    test_all_time_list = get_test_time_list_2023()
    batch_size = data_config['batch_frames'] 
    pixel_step = data_config['batch_pixels'] 
    time_num = len(test_all_time_list)
    era5_example = get_era5_data(test_all_time_list[0])
    pixels_num = era5_example.shape[0]
    im_ch =1
    obs_pos = wind_n_sensors(era5_example, 
                             data_config['num_sensors'], 
                             data_config['seed']) 
    obs_coords = era5_gfs_pos_encoding[obs_pos,].to(device) 
    era5_mask = (era5_example == 0).reshape(201, 241, 1)
    era5_mask = torch.tensor(era5_mask, dtype=torch.bool)
    output_im = torch.zeros(time_num, pixels_num, im_ch).to(device)

    with torch.no_grad():
        for i in tqdm(range(0, time_num, batch_size)):
            batch_times = test_all_time_list[i:i+batch_size]
            length_all = []
            sensor_value_all = []
            gfs_data_all = []
            for j in range(len(batch_times)):
                time_sel = batch_times[j]
                era5_sel = get_era5_data(time_sel) 
                gfs_sel = get_gfs_data(time_sel) 
                obs_sel = era5_sel[obs_pos,].to(device)
                obs_num = obs_pos.shape[0]
                obs_value = torch.cat([obs_sel, obs_coords], dim=-1) 

                length_all.append(obs_num) 
                sensor_value_all.append(obs_value) 
                gfs_data_all.append(gfs_sel)  

            gfs_data_all = torch.stack(gfs_data_all, dim=0).to(device)
            max_len = max(length_all)
            feature_dim = 129
            padded_obs = torch.zeros(len(batch_times), max_len, feature_dim).to(device) 
            pad_masks = torch.ones(len(batch_times), max_len, dtype=torch.bool).to(device)
            for k in range(len(length_all)):
                padded_obs[k, :length_all[k]] = sensor_value_all[k]  
                pad_masks[k, :length_all[k]] = False  

            all_pixel_indices = np.arange(pixels_num)
            for m in range(0, pixels_num, pixel_step):
                pixel_indices = all_pixel_indices[m:m+pixel_step]
                gfs_field_values = gfs_data_all[:, pixel_indices,]  
                gfs_pos = era5_gfs_pos_encoding[pixel_indices,]  
                gfs_pos = gfs_pos.unsqueeze(0).expand(gfs_field_values.shape[0], -1, -1)
                gfs_coords = torch.cat([gfs_field_values, gfs_pos], dim=-1).to(device) 
                output = gfs_field_values + model(padded_obs, pad_masks, gfs_coords)  
                output_im[i:i+len(batch_times), m:m+len(pixel_indices)] = output

        output_im = output_im.reshape(-1, 201, 241, 1)
        full_mask = era5_mask.expand(output_im.shape[0], -1, -1, -1).to(device)  
        output_im[full_mask] = 0
        torch.save(output_im.cpu(), f'{save_path}/res.torch')

if __name__=='__main__':
    main()


