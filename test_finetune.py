

import numpy as np
from glob import glob as gb
import torch
from s_parser import parse_args
from network import SwiftWind
import time
import os
import pandas as pd
import math

def get_test_time_list_2023():
    start = np.datetime64('2023-01-01T00:00:00')
    end = np.datetime64('2023-12-31T23:00:00')
    step = np.timedelta64(60, 'm')
    time_series = np.arange(start, end + step, step, dtype='datetime64[ns]')
    time_str_list = pd.to_datetime(time_series).strftime('%Y%m%dT%H%M%S')
    return time_str_list

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

def get_gfs_era5_lonlat():
    lons = np.arange(90, 150.01, 0.25)  
    lats = np.arange(40, -10.01, -0.25)   
    lon_grid, lat_grid = np.meshgrid(lons, lats)  
    latlon_flat = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=1)  
    latlon_tensor = torch.tensor(latlon_flat, dtype=torch.float32)  
    all_pos_encoding = latlon_fourier_encoding(latlon_tensor, num_bands=32)
    return all_pos_encoding

def get_era5_data(time_sel):
    era5_dir = './final_data_used/used_model_all_data/era5_1h_npy_2017_2023'
    era5_name = f'era5_{time_sel}.npy'
    era5_path = os.path.join(era5_dir, era5_name)
    era5_data = np.load(era5_path)
    era5_data = np.nan_to_num(era5_data)
    return era5_data.reshape(-1, 1)

def get_gfs_data(time_sel):
    gfs_dir = './final_data_used/used_model_all_data/gfs_1h_npy_2017_2023'
    gfs_name = f'gfs_{time_sel}.npy'
    gfs_path = os.path.join(gfs_dir, gfs_name)
    gfs_data = np.load(gfs_path)
    gfs_data = np.nan_to_num(gfs_data)
    return torch.tensor(gfs_data.reshape(-1, 1), dtype=torch.float32) 

def read_npy_data(file_path):
    loaded = np.load(file_path, allow_pickle=True).item()
    lat = np.array(loaded['lat'])
    lon = np.array(loaded['lon'])
    ws = np.array(loaded['wind_speed'])
    norm_mean = 5.66179499128632
    norm_std = 3.024306291995
    ws = (ws - norm_mean) / norm_std
    return lon.reshape(-1,1), lat.reshape(-1,1), ws.reshape(-1,1)
        
def get_downsample_obs(time_sel):
    obs_down_dir = './data_reprocess/downsample_obs_all'
    obs_down_name = f'{time_sel}_obs_downsample.npy'
    obs_down_path = os.path.join(obs_down_dir, obs_down_name)
    obs_down_lon, obs_down_lat, obs_down_ws = [torch.tensor(x, dtype=torch.float32) for x in read_npy_data(obs_down_path)]
    obs_down_pos_encoding = latlon_fourier_encoding(torch.stack([obs_down_lat, obs_down_lon], dim=1).squeeze(), num_bands=32)
    obs_down_sensor_values = torch.cat([obs_down_ws, obs_down_pos_encoding], dim=-1)
    return obs_down_sensor_values

# 获取ascat1h的数据
def get_ascat_1h_init_background(time_sel):
    ascat_1h_dir = './data_reprocess/ascat_1h_interp_2023'
    ascat_1h_name = f'{time_sel}_ascat_1h_interp_2023.npy'
    ascat_1h_path = os.path.join(ascat_1h_dir, ascat_1h_name)
    loaded = np.load(ascat_1h_path, allow_pickle=True).item()
    lat = np.array(loaded['lat'])
    lon = np.array(loaded['lon'])
    gfs_interp_ws = np.array(loaded['gfs_interp_ws'])
    norm_mean = 5.66179499128632
    norm_std = 3.024306291995
    gfs_interp_ws = (gfs_interp_ws - norm_mean) / norm_std
    return lon.reshape(-1,1), lat.reshape(-1,1), gfs_interp_ws.reshape(-1,1)

def main():
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

    start_time = time.time()
    save_path = model_loc.split('checkpoints')[0]
    era5_gfs_pos_encoding = get_gfs_era5_lonlat().to(device)  
    test_all_time_list = get_test_time_list_2023()
    mask_select = np.load("/home/bingxing2/home/scx6a12/data_reprocess/obs_select_mask_500_1000.npy")
    mask_used = mask_select[52562:61322]
    test_all_time_list = test_all_time_list[mask_used]
    
    batch_size = 128
    pixel_step = 8192
    time_num = len(test_all_time_list)
    era5_example = get_era5_data(test_all_time_list[0])
    im_ch =1
    era5_mask = (era5_example == 0).reshape(201, 241, 1)
    era5_mask = torch.tensor(era5_mask, dtype=torch.bool)
    output_all = []
    with torch.no_grad():
        for i in range(time_num):
            time_sel = test_all_time_list[i]
            obs_sensor_values = get_downsample_obs(time_sel).to(device) 
            padded_obs = obs_sensor_values.unsqueeze(0).to(device) 
            pad_masks = torch.zeros(1, obs_sensor_values.shape[0], dtype=torch.bool).to(device)
            query_lon, query_lat, query_init_value = get_ascat_1h_init_background(time_sel) 
            query_init_value = torch.tensor(query_init_value, dtype=torch.float32)
            query_pos_encoding = latlon_fourier_encoding(torch.stack([torch.tensor(query_lat, dtype=torch.float32), torch.tensor(query_lon, dtype=torch.float32)], dim=1).squeeze(-1), num_bands=32) # [M,128]
            query_sensor_values = torch.cat([query_init_value, query_pos_encoding], dim=-1).unsqueeze(0).to(device) # [1,M,129]

            back = query_init_value.unsqueeze(0).to(device)
            output = back + model(padded_obs, pad_masks, query_sensor_values)  
            output_all.append(output.cpu())
            print(f'{time_sel}_output.shape:{output.shape},query_lon.shape:{query_lon.shape}')

        torch.save(output_all, f'{save_path}/ascat_1h_finetune.torch')
        compute_end_time = time.time() - start_time

if __name__=='__main__':
    main()










