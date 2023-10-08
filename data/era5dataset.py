import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon

import torch
import os
from torch.utils.data import TensorDataset, DataLoader
import lightning as pl
import matplotlib.pyplot as plt
import xarray as xr

def get_era5(data_dir, label_key, remove_nans=True, normalize=True):
    
    if not os.path.exists(data_dir):
        print(f"{data_dir} does not exist")
        
    fp = os.path.join(data_dir,  "era5.nc")
    if not os.path.exists(fp):
        print(f"{fp} does not exist")
        
        
    ds = xr.open_dataset(fp)
    ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180)  
    df = ds.to_dataframe().reset_index()
    if type(label_key) is list:
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))[["longitude", "latitude"] + label_key]
    else:
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))[["longitude", "latitude", label_key]]
    
    if normalize:
        if type(label_key) is list:
            for key in label_key:
                gdf[key] = (( gdf[key] -  gdf[key].min()) / ( gdf[key].max() -  gdf[key].min()))
        else:
            gdf[label_key] = (( gdf[label_key] -  gdf[label_key].min()) / ( gdf[label_key].max() -  gdf[label_key].min()))

    if remove_nans:
        gdf = gdf.dropna(subset=label_key)
     
    return gdf

def split_era5_dataframe(df, label_key, random_seed=0 ,train_size=0.01,val_size=0.05):
    rs = np.random.RandomState(random_seed)
    
    n = 6483600
    train_split_size = int(n * train_size)
    val_split_size = int(n * val_size)
    test_split_size = n - train_split_size - val_split_size
    n_to_sample = train_split_size + val_split_size + test_split_size
    
    assert np.sum(n_to_sample) <= len(df)
    
    shuffle_list = rs.choice(len(df), len(df), replace=False)
    
    shuffled_df = df.iloc[shuffle_list].reset_index(drop=True)
    
    n1 = train_split_size
    n2 = train_split_size + val_split_size
    
    df_train = shuffled_df.iloc[0:n1].reset_index(drop=True)
    df_val = shuffled_df.iloc[n1: n2].reset_index(drop=True)
    df_test = shuffled_df.iloc[n2: n_to_sample].reset_index(drop=True)
    
    data_by_split = {}
    for split, df_split in zip(['train', 'val', 'test'], [df_train, df_val, df_test]):
        locs = torch.Tensor(df_split[['longitude','latitude']].values) 
        if type(label_key) is list:
            labels = torch.Tensor(df_split[label_key].values).double()   
        else:
            labels = torch.Tensor(df_split[label_key].values).double().unsqueeze(-1)        
        data_by_split[split] = [locs, labels]

    return data_by_split
        
def get_era5_data_by_split(data_root,label_key="t2m",random_seed=0):
    era5_df = get_era5(data_root,label_key=label_key)
    return split_era5_dataframe(era5_df, label_key=label_key, random_seed=random_seed)
    
class ERA5DataModule(pl.LightningDataModule):
    def __init__(self, num_workers=0, batch_size=1000,data_root='/home/kklemmer/sphericalharmonics/data/era5', label_key='t2m'):
        super().__init__()
        self.batch_size=batch_size
        self.data_root=data_root
        self.num_workers=num_workers
        self.label_key = label_key
        

    def setup(self, stage: str):
        data_by_split = get_era5_data_by_split(data_root=self.data_root, label_key=self.label_key, random_seed=0)
        self.train_ds = TensorDataset(*data_by_split['train'])
        self.valid_ds = TensorDataset(*data_by_split['val'])
        self.evalu_ds = TensorDataset(*data_by_split['test'])

        self.test_locs = data_by_split['test'][0].detach()
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.evalu_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def get_test_locs(self):
        return self.test_locs
    
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
    
#     # get data
#     DATA_ROOT = '/home/esther/sphericalharmonics/data/sea_ice/'
#     data_by_split = get_data_by_split(data_root=DATA_ROOT)
#     test_locs, test_labels = data_by_split['test']
#     df = pd.DataFrame(test_locs)
#     gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="4326")
#     gdf['ice_thickness_int'] = test_labels

    
#     # Set the coordinate reference system (CRS) to arctic region
#     crs = '3413'
#     gdf_arctic = gdf.to_crs(epsg=crs)  

#     coastlines = gpd.read_file(os.path.join(DATA_ROOT, 'sea_ice','ne_10m_coastline.shp')  )
#     coastlines_arctic = coastlines.to_crs(epsg=crs) 
    
#     #plot
#     fig, ax = plt.subplots(figsize=(8,8))  
#     p = gdf_arctic.plot('ice_thickness_int', ax=ax, markersize=4, legend=True, cmap='cool_r',
#                    legend_kwds={'shrink':0.6}, vmin=0, vmax=6.5)
#     # lims before plotting coastline so we can revert to this
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#     coastlines_arctic.plot(ax=ax, edgecolor='black', linewidth=0.5)  

    # # Set the extent of the plot to cover the Arctic area  
    # ax.set_xlim(*xlim);  
    # ax.set_ylim(*ylim); 

    # ax.set_title('test set points: interpolated ice thickness (m)')
    # ax.axis("off")
    # plt.tight_layout()
    # plt.show()
    

    # data_figs_dir = os.path.join(DATA_ROOT, 'figs')
    # if not os.path.exists(data_figs_dir):
    #     os.mkdir(data_figs_dir)
    # # save
    # fig.savefig(os.path.join(data_figs_dir, 'seaicedataset.png'), 
    #             transparent=True, bbox_inches="tight", 
    #             pad_inches=0)