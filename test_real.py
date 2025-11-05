
# test.py for Real-world observational experiments

import numpy as np
from glob import glob as gb
import torch
from s_parser import parse_args
from network import SwiftWind
from tqdm import tqdm
import os
import pandas as pd
import math

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
        latlon_rad = latlon_tensor * math.pi / 180.0  # [N, 2]
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

# get ERA5 data
def get_era5_data(time_sel):
    era5_dir = './final_data_used/used_model_all_data/era5_1h_npy_2017_2023'
    era5_name = f'era5_{time_sel}.npy'
    era5_path = os.path.join(era5_dir, era5_name)
    era5_data = np.load(era5_path)
    era5_data = np.nan_to_num(era5_data)
    return era5_data.reshape(-1, 1)

# get GFS-6h data
def get_gfs_data(time_sel):
    gfs_dir = './final_data_used/used_model_all_data/gfs_1h_npy_2017_2023'
    gfs_name = f'gfs_{time_sel}.npy'
    gfs_path = os.path.join(gfs_dir, gfs_name)
    gfs_data = np.load(gfs_path)
    gfs_data = np.nan_to_num(gfs_data)
    return torch.tensor(gfs_data.reshape(-1, 1), dtype=torch.float32) 

# get simple timestep data
def read_npy_data(file_path):
    loaded = np.load(file_path, allow_pickle=True).item()
    lat = np.array(loaded['lat'])
    lon = np.array(loaded['lon'])
    ws = np.array(loaded['era5_interp_ws'])
    norm_mean = 5.66179499128632
    norm_std = 3.024306291995
    ws = (ws - norm_mean) / norm_std
    return lon.reshape(-1,1), lat.reshape(-1,1), ws.reshape(-1,1)
        
# get downsampled observational data 
# differnet configurations: all/selected, idealized/real
def get_downsample_obs(time_sel):
    obs_down_dir = './data_reprocess/interp_downsample_obs_all'
    obs_down_name = f'{time_sel}_obs_downsample_interp.npy'
    obs_down_path = os.path.join(obs_down_dir, obs_down_name)
    obs_down_lon, obs_down_lat, obs_down_ws = [torch.tensor(x, dtype=torch.float32) for x in read_npy_data(obs_down_path)]
    obs_down_pos_encoding = latlon_fourier_encoding(torch.stack([obs_down_lat, obs_down_lon], dim=1).squeeze(), num_bands=32)
    obs_down_sensor_values = torch.cat([obs_down_ws, obs_down_pos_encoding], dim=-1)
    return obs_down_sensor_values


def main():
    # initial model
    data_config, encoder_config, decoder_config, train_config = parse_args()
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
    mask_select = np.load("./data_reprocess/obs_select_mask_500_1000.npy")
    mask_used = mask_select[52562:61322]
    test_all_time_list = test_all_time_list[mask_used]
    
    batch_size = 128
    pixel_step = 8192
    time_num = len(test_all_time_list)
    era5_example = get_era5_data(test_all_time_list[0])
    pixels_num = era5_example.shape[0]
    im_ch =1
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
                obs_sensor_values = get_downsample_obs(time_sel).to(device)
                length_all.append(obs_sensor_values.shape[0])
                sensor_value_all.append(obs_sensor_values)
                gfs_data_all.append(get_gfs_data(time_sel))  
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










