
# dataloaders.py for arbitrary-location queries

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import math
import pandas as pd

# get simple timestep data
def read_npy_data(file_path):
    loaded = np.load(file_path, allow_pickle=True).item()
    lat = np.array(loaded['lat'])
    lon = np.array(loaded['lon'])
    ws = np.array(loaded['wind_speed'])
    norm_mean = 5.66179499128632
    norm_std = 3.024306291995
    ws = (ws - norm_mean) / norm_std
    return lon.reshape(-1,1), lat.reshape(-1,1), ws.reshape(-1,1)

# lon-lat positional encoding
def latlon_fourier_encoding(latlon_tensor, num_bands=32):
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

# get downsampled observational data 
# differnet configurations: all/selected, idealized/real
def get_downsample_obs(time_sel):
    obs_down_dir = './scx6a12/data_reprocess/downsample_obs_all'
    obs_down_name = f'{time_sel}_obs_downsample.npy'
    obs_down_path = os.path.join(obs_down_dir, obs_down_name)
    obs_down_lon, obs_down_lat, obs_down_ws = [torch.tensor(x, dtype=torch.float32) for x in read_npy_data(obs_down_path)]
    obs_down_pos_encoding = latlon_fourier_encoding(torch.stack([obs_down_lat, obs_down_lon], dim=1).squeeze(), num_bands=32)
    obs_down_sensor_values = torch.cat([obs_down_ws, obs_down_pos_encoding], dim=-1)
    return obs_down_sensor_values, obs_down_sensor_values.shape[0]

# get ERA5 data 
def get_era5_data(time_sel):
    era5_dir = './final_data_used/used_model_all_data/era5_1h_npy_2017_2023'
    era5_name = f'era5_{time_sel}.npy'
    era5_path = os.path.join(era5_dir, era5_name)
    era5_data = np.load(era5_path)
    era5_data = np.nan_to_num(era5_data)
    return torch.tensor(era5_data.reshape(-1, 1), dtype=torch.float32)

# get GFS-6h data
def get_gfs_data(time_sel):
    gfs_dir = './final_data_used/used_model_all_data/gfs_1h_npy_2017_2023'
    gfs_name = f'gfs_{time_sel}.npy'
    gfs_path = os.path.join(gfs_dir, gfs_name)
    gfs_data = np.load(gfs_path)
    gfs_data = np.nan_to_num(gfs_data)
    return torch.tensor(gfs_data.reshape(-1, 1), dtype=torch.float32) 

# get all timestep list of 1h-resolution and delete the inaccurate ERA5 steps
def get_used_1h_time_list():
    start = np.datetime64('2017-01-01T00:00:00')
    end = np.datetime64('2023-12-31T23:50:00')
    step = np.timedelta64(60, 'm')
    time_series = np.arange(start, end, step, dtype='datetime64[ns]')
    time_str_list = pd.to_datetime(time_series).strftime('%Y%m%dT%H%M%S')

    GFS_delete = [12264,12265,12266,12267,12268,12269]
    ERA5_delete = [6810,6811,6812,6813,6814,14922,14923,14924,14925,25191,25192,25193,25194,25195,25196,25197]
    all_delete_index = GFS_delete + ERA5_delete
    mask = np.ones(len(time_str_list), dtype=bool)
    mask[all_delete_index] = False
    filtered_time_str_list = time_str_list[mask]
    return filtered_time_str_list

# get 1h-ascat data
def get_ascat_1h_init_background(time_sel):
    ascat_1h_dir = './data_reprocess/ascat_1h_interp_2017_2023'
    ascat_1h_name = f'{time_sel}_ascat_1h_interp.npy'
    ascat_1h_path = os.path.join(ascat_1h_dir, ascat_1h_name)
    loaded = np.load(ascat_1h_path, allow_pickle=True).item()
    lat = np.array(loaded['lat'])
    lon = np.array(loaded['lon'])
    ws = np.array(loaded['wind_speed'])
    era5_interp_ws = np.array(loaded['era5_interp_ws'])
    gfs_interp_ws = np.array(loaded['gfs_interp_ws'])
    norm_mean = 5.66179499128632
    norm_std = 3.024306291995
    ws = (ws - norm_mean) / norm_std
    gfs_interp_ws = (gfs_interp_ws - norm_mean) / norm_std
    if ws.shape[0] > 2000:
        ascat_lat = torch.tensor(lat, dtype=torch.float32)
        ascat_lon = torch.tensor(lon, dtype=torch.float32)
        ascat_pos_encoding = latlon_fourier_encoding(torch.stack([ascat_lat, ascat_lon], dim=1).squeeze(), num_bands=32)
        gfs_interp_ws_value = torch.tensor(gfs_interp_ws, dtype=torch.float32)
        ascat_ws = torch.tensor(ws.reshape(-1,1), dtype=torch.float32)
        ascat_sensor_values = torch.cat([gfs_interp_ws_value, ascat_pos_encoding], dim=-1)
        return ascat_sensor_values, ascat_ws 
    else:
        return torch.empty((0, 129), dtype=torch.float32), torch.empty((0, 1), dtype=torch.float32)


# dataloaders for train and validate
def SwiftWind_train_val_dataloaders(data_config, num_workers=4):
    # train
    train_ds = SwiftWindDataset(data_config, start_ind=0, end_ind=48146)
    train_loader = DataLoader(
        train_ds, 
        batch_size=data_config['batch_frames'], 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    # validate
    val_ds = SwiftWindDataset(data_config, start_ind=48146, end_ind=52562)
    val_loader = DataLoader(
        val_ds, 
        batch_size=data_config['batch_frames'], 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    return train_loader, val_loader

# deal length-variant input batch data and get the mask array
def custom_collate_fn(batch):
    B = len(batch)
    lengths = [x[0].shape[0] for x in batch]        
    max_len = max(lengths)                           
    feature_dim = batch[0][0].shape[1]                

    padded_icoads = torch.zeros(B, max_len, feature_dim, dtype=batch[0][0].dtype)
    pad_masks = torch.ones(B, max_len, dtype=torch.bool)  

    gfs_coords_tensor = torch.stack([x[2] for x in batch])     
    era5_field_tensor = torch.stack([x[3] for x in batch])     

    for i, (sensor_values, icoads_num, _, _) in enumerate(batch):
        cur_len = sensor_values.shape[0]
        padded_icoads[i, :cur_len] = sensor_values           
        pad_masks[i, :cur_len] = False                       

    return padded_icoads, pad_masks, gfs_coords_tensor, era5_field_tensor

class SwiftWindDataset(Dataset):
    def __init__(self, data_config, start_ind, end_ind):
        data_config['image_size']   = [201, 241]
        data_config['im_ch']        = 1

        mask_select = np.load("./data_reprocess/obs_select_mask_500_1000.npy")
        mask_used = mask_select[start_ind:end_ind]

        self.era5_gfs_pos_encoding = get_gfs_era5_lonlat()  
        self.all_time_str_list_init = get_used_1h_time_list()[start_ind:end_ind]
        self.all_time_str_list = self.all_time_str_list_init[mask_used]
        self.N_time = len(self.all_time_str_list)

        era5_example = get_era5_data(time_sel=self.all_time_str_list[0])
        self.pix_avail = era5_example.nonzero()[:,0] 
        self.N_pixel = len(self.pix_avail)

        self.batch_frames = data_config['batch_frames'] 
        self.batch_pixels = data_config['batch_pixels']
        self.pixel_batches_per_time = int(np.ceil(self.N_pixel / self.batch_pixels))
        
        self.index_pairs = []
        for t_idx in range(self.N_time):
            for p_idx in range(self.pixel_batches_per_time):
                self.index_pairs.append((t_idx, p_idx))

        data_config['num_batches'] = len(self.index_pairs)
        print(">>> Dataset __init__ called")

    def __len__(self):
        return len(self.index_pairs)
    
    def __getitem__(self, idx):
        time_idx, pixel_group_idx = self.index_pairs[idx]
        time_sel = self.all_time_str_list[time_idx]

        obs_sensor_values, obs_num = get_downsample_obs(time_sel)
        ascat_coords_2, ascat_field_values2 = get_ascat_1h_init_background(time_sel) 
        ascat_length = ascat_field_values2.shape[0]

        if ascat_length < self.batch_pixels:
            gfs_length = self.batch_pixels - ascat_length
            idxs = torch.randint(0, self.N_pixel, (self.batch_pixels,))
            random_pixels = self.pix_avail[idxs][0:gfs_length]
            gfs_coords_1 = torch.cat([get_gfs_data(time_sel)[random_pixels,], 
                                        self.era5_gfs_pos_encoding[random_pixels,]],
                                        dim=-1) 
            era5_field_values1 = get_era5_data(time_sel)[random_pixels,]  

            query_coords = torch.cat([ascat_coords_2, gfs_coords_1], dim=0)
            ref_field_values = torch.cat([ascat_field_values2, era5_field_values1], dim=0)
        else:
            perm_a = torch.randperm(ascat_length)[:self.batch_pixels]
            query_coords = ascat_coords_2[perm_a]         
            ref_field_values = ascat_field_values2[perm_a] 

        return obs_sensor_values, obs_num, query_coords, ref_field_values























