import geopandas
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data import TensorDataset, DataLoader
import lightning as pl

DATA_DIR = "datasets"
os.makedirs(DATA_DIR, exist_ok=True)

# if not present, please download
# "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/physical/ne_50m_land.zip"
# an unzip all files (.shp and others) to
SHAPEFILEPATH = "datasets/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def get_data_points(N=5000, seed=0, cache=True, sphericaluniform=True, grid=False):
    if grid:
        cachefilename = DATA_DIR + f"/landoceansdatasetgrid.geojson"
    else:
        cachefilename = DATA_DIR + f"/landoceansdataset{seed}.geojson"
    if os.path.exists(cachefilename) and cache:
        print(f"reading dataset from {cachefilename}. delete file to regenerate...")
        points = gpd.read_file(cachefilename)
    else:
        print(f"generating {cachefilename} from {SHAPEFILEPATH}")
        world = geopandas.read_file(SHAPEFILEPATH).dissolve()
        #world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres')).to_crs(4326).dissolve()

        rng = np.random.RandomState(seed)
        if grid:
            gridsize = 1e5 # in m
            x, y = np.meshgrid(np.arange(-17367530.45, 17367530.45, step=gridsize), np.arange(-7324184.56, 7324184.56, step=gridsize))
            geometry = gpd.points_from_xy(x.reshape(-1), y.reshape(-1), crs=6933).to_crs(4326)

            lons = geometry.x
            lats = geometry.y
        elif sphericaluniform:
            x, y, z = rng.normal(size=(3, N))
            az, el, _ = cart2sph(x, y, z)
            lons, lats = np.rad2deg(az), np.rad2deg(el)
        else:
            lats = (rng.rand(N) * 180) - 90
            lons = (rng.rand(N) * 360) - 180

        points = gpd.GeoDataFrame([],geometry=[Point(lon, lat) for lat, lon in zip(lats, lons)], crs=4326)
        points["land"] = [bool(world.contains(pt.geometry).any()) for idx, pt in points.iterrows()]

        points["theta"] = points.geometry.x.apply(np.deg2rad) + np.pi * 2
        points["phi"] = points.geometry.y.apply(np.deg2rad) + np.pi/2

        points.to_file(cachefilename)

    return points

def get_data(N=5000, seed=0, grid=False):
    points = get_data_points(N, seed, grid=grid)

    lon = torch.tensor(points.geometry.x.values)
    lat = torch.tensor(points.geometry.y.values)

    lonlats = torch.stack([lon, lat], dim=1)

    land = points.land.astype(float).values  # -.5 for ocean +0.5 for land
    land = torch.from_numpy(land).unsqueeze(-1)

    return lonlats, land

class LandOceanDataModule(pl.LightningDataModule):
    def __init__(self, num_samples=5000, batch_size=1000):
        super().__init__()
        self.num_samples = num_samples
        self.batch_size=batch_size

    def setup(self, stage: str):
        self.train_ds = TensorDataset(*get_data(self.num_samples, seed=0))
        self.valid_ds = TensorDataset(*get_data(self.num_samples, seed=1))
        self.evalu_ds = TensorDataset(*get_data(self.num_samples, grid=True))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.evalu_ds, batch_size=self.batch_size, shuffle=False)


if __name__ == '__main__':
    points = get_data_points(seed=2, cache=False, grid=True)

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres')).to_crs(4326).dissolve()

    fig, ax = plt.subplots(figsize=(16,9))
    world.plot(ax=ax, color="black")

    color = "red"
    ax.scatter(points.loc[points.land].geometry.x, points.loc[points.land].geometry.y, color="red", marker="*")
    ax.scatter(points.loc[~points.land].geometry.x, points.loc[~points.land].geometry.y, color="blue", marker="+")
    ax.axis("off")
    points.head()
    plt.tight_layout()
    plt.show()

    print(points.land.sum(), (len(points) - points.land.sum()))

    fig.savefig(DATA_DIR+"/landoceansdataset.png", transparent=True, bbox_inches="tight", pad_inches=0)
