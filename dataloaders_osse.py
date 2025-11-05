
# dataloaders.py for Observing system simulation experiments (OSSEs)

import os
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
import math
import pandas as pd

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
    """
    batch: List[Tuple[
        icoads_sensor_values: Tensor[Ni, 129],
        icoads_num: int,
        gfs_coords_new: Tensor[2048, 129],
        era5_field_values: Tensor[2048, 1]
    ]]
    
    Returns:
        padded_icoads: Tensor[B, max_len, 129]
        pad_masks:     BoolTensor[B, max_len] 
        gfs_coords_tensor: Tensor[B, 2048, 129]
        era5_field_tensor: Tensor[B, 2048, 1]
    """

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

        self.era5_gfs_pos_encoding = get_gfs_era5_lonlat() 
        self.all_time_str_list  = get_used_1h_time_list()[start_ind:end_ind]
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
        self.obs_pos = wind_n_sensors(data=era5_example, 
                                      n_sensors=data_config['num_sensors'], 
                                      rnd_seed=data_config['seed'])
        
        print(">>> Dataset __init__ called")

    def __len__(self):
        return len(self.index_pairs)
    
    def __getitem__(self, idx):
        time_idx, pixel_group_idx = self.index_pairs[idx]
        time_sel = self.all_time_str_list[time_idx]
        idxs = torch.randint(0, self.N_pixel, (self.batch_pixels,))
        random_pixels = self.pix_avail[idxs]

        obs_num = self.obs_pos.shape[0]
        gfs_coords_new = torch.cat([get_gfs_data(time_sel)[random_pixels,], 
                                    self.era5_gfs_pos_encoding[random_pixels,]],
                                    dim=-1) 
        obs_sensor_values = torch.cat([get_era5_data(time_sel)[self.obs_pos,],
                                       self.era5_gfs_pos_encoding[self.obs_pos,]],
                                       dim=-1) 
        era5_field_values = get_era5_data(time_sel)[random_pixels,]
        return obs_sensor_values, obs_num, gfs_coords_new, era5_field_values









