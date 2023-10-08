import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl

"""
fibonacci latice following this implementation
https://arxiv.org/pdf/0912.4540.pdf
"""

def generate_fibonaccilattice(N, n_classes=16):
    assert 4 % 2 == 0, "Fibonacci Lattice does only work with an even number of points"

    N = N // 2

    import math
    # golden ratio
    phi = (1 + math.sqrt(5)) / 2

    lats, lons, labels = [], [], []

    for i in np.arange(-N, N):
        lat = np.arcsin( (2*i) / (2*N + 1) ) * 180 / np.pi
        lon = (i % phi) * (360 / phi)

        if lon < -180:
            lon += 360
        if lon > 180:
            lon -= 360

        lons.append(lon)
        lats.append(lat)
        labels.append(i % n_classes)

    return np.stack(lons), np.stack(lats), np.stack(labels)

def sph2cart(longitude, latitude, radius=1):
    """
    Converts spherical coordinates (longitude, latitude) to 3D Cartesian coordinates (x, y, z).

    Args:
        longitude (float or numpy.ndarray): The longitude(s) in radians.
        latitude (float or numpy.ndarray): The latitude(s) in radians.
        radius (float or numpy.ndarray): The radius(s) representing the distance from the origin.

    Returns:
        numpy.ndarray: An array of shape (3,) or (N, 3) containing the Cartesian coordinates (x, y, z).
    """
    x = radius * np.cos(latitude) * np.cos(longitude)
    y = radius * np.cos(latitude) * np.sin(longitude)
    z = radius * np.sin(latitude)
    return np.column_stack((x, y, z))

def pairwise_euclidean_distance(matrix1, matrix2):
    """
    Calculates the pairwise Euclidean distances between each point pair in two matrices.

    Args:
        matrix1 (numpy.ndarray): The first matrix of shape (N1, 3).
        matrix2 (numpy.ndarray): The second matrix of shape (N2, 3).

    Returns:
        numpy.ndarray: A matrix of shape (N1, N2) containing the pairwise Euclidean distances.
    """
    diff = matrix1[:, np.newaxis] - matrix2
    distances = np.linalg.norm(diff, axis=-1)
    return distances


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def haversine_distance(lon1, lat1, lon2, lat2, radius=1.0):
    """
    Calculates the pairwise Haversine distances between each point pair.

    Args:
        lon1 (numpy.ndarray): The longitudes of the first set of points, shape (N1,).
        lat1 (numpy.ndarray): The latitudes of the first set of points, shape (N1,).
        lon2 (numpy.ndarray): The longitudes of the second set of points, shape (N2,).
        lat2 (numpy.ndarray): The latitudes of the second set of points, shape (N2,).
        radius (float, optional): The radius representing the distance from the origin. Defaults to 1.0.

    Returns:
        numpy.ndarray: A matrix of shape (N1, N2) containing the pairwise Haversine distances.
    """
    lon1, lat1, lon2, lat2 = np.radians(lon1), np.radians(lat1), np.radians(lon2), np.radians(lat2)
    dlon = lon2[:, np.newaxis] - lon1
    dlat = lat2[:, np.newaxis] - lat1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2[:, np.newaxis]) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    distances = radius * c
    return distances

def assign_closest_label(lons_grid, lats_grid, lons, lats, labels):
    """
    assigns the label of lons, lats, to the closest lons_grid, lats_grid point
    """
    distance_matrix = haversine_distance(lons_grid, lats_grid, lons, lats)
    return labels[distance_matrix.argmin(0)]

def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    import matplotlib.patheffects as path_effects

    cmap = 'RdBu'
    alpha = .8
    size = 15

    lons, lats, labels = generate_fibonaccilattice(N=100, n_classes=16)
    lons_grid, lats_grid, _ = generate_fibonaccilattice(10000)

    distance_matrix = haversine_distance(lons_grid, lats_grid, lons, lats)
    labels_grid = labels[distance_matrix.argmin(0)]

    fig = plt.figure(figsize=(3,3))
    map = Basemap(projection='ortho', lat_0=45, lon_0=30, resolution='l')


    map.scatter(lons_grid, lats_grid, s=size, c=labels_grid, latlon=True, cmap=cmap, alpha=alpha, edgecolor="none")
    #map.scatter(lons, lats, c="white", s=200, latlon=True)

    map.fillcontinents(color="black", alpha=0.33)
    map.drawcoastlines()

    ax = plt.gca()
    X, Y = map(lons, lats)
    for id, (x, y, l) in enumerate(zip(X,Y, labels)):
        if not (np.isinf(x) or np.isinf(x)):
            text = ax.text(x, y, str(l), ha="center", va="center", color="black")
            text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                                   path_effects.Normal()])


    plt.tight_layout()
    fig.savefig("/tmp/checkerboardsphere.pdf", transparent=True, bbox_inches="tight", pad_inches=0)
    fig.savefig("/tmp/checkerboardsphere.png", transparent=True, bbox_inches="tight", pad_inches=0)

    fig = plt.figure()
    map = Basemap()


    map.scatter(lons_grid, lats_grid, s=size, c=labels_grid, latlon=True, cmap=cmap, alpha=alpha, edgecolor="none")
    # map.scatter(lons, lats, c="white", s=200, latlon=True)

    map.fillcontinents(color="black", alpha=0.33)
    map.drawcoastlines()

    ax = plt.gca()
    X, Y = map(lons, lats)
    for id, (x, y, l) in enumerate(zip(X, Y, labels)):
        if not (np.isinf(x) or np.isinf(x)):
            text = ax.text(x, y, str(l), ha="center", va="center", color="black")
            text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                                   path_effects.Normal()])

    ax = plt.gca()
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    plt.tight_layout()
    fig.savefig("/tmp/checkerboardmap.pdf", transparent=True, bbox_inches="tight", pad_inches=0)
    fig.savefig("/tmp/checkerboardmap.png", transparent=True, bbox_inches="tight", pad_inches=0)

    plt.show()

def calculate_average_distance_between_closest_neighbors(lons, lats, unit="km"):
    if unit=="km":
        radius = 6371
    elif unit=="m":
        radius = 6371000
    elif unit == "rad" or unit == "deg":
        radius = 1

    distances = haversine_distance(lons, lats, lons, lats, radius=radius)

    # remove distance of points to itself
    np.fill_diagonal(distances, np.inf)

    mean_distance = distances.min(1).mean()
    std_distance = distances.min(1).std()

    if unit == "deg":
        return np.rad2deg(mean_distance), np.rad2deg(std_distance)
    else:
        return mean_distance, std_distance

def area_per_point(N, r=6371):
    return 4 * np.pi * r ** 2 / N

def resolution_deg(N):
    # approximates the average distance between points in degree
    # idea: equal area per point: area/points -> radians of a euclidean circle (approximation) -> in degree
    # this slightly underestimates the radius, due to the euclidean circle but within
    np.rad2deg(np.sqrt(area_per_point(N, r=1) / np.pi))

def calc_avg_distances(N, unit="deg"):
    lons, lats, labels = generate_fibonaccilattice(N)
    return calculate_average_distance_between_closest_neighbors(lons, lats, unit=unit)

def get_data(N_samples, N_support, n_classes, seed=0, grid=False, random_classes=False):
    lons, lats, labels = generate_fibonaccilattice(N_support, n_classes=n_classes)

    if random_classes:
        labels = (np.random.RandomState(seed).rand(len(labels)) * n_classes).astype(int)

    if grid:
        lons_grid, lats_grid, _ = generate_fibonaccilattice(N_samples)
        labels_grid = assign_closest_label(lons_grid, lats_grid, lons, lats, labels)

        lonlats = torch.from_numpy(np.stack([lons_grid, lats_grid])).T
        labels = torch.from_numpy(labels_grid)

    else:
        rng = np.random.RandomState(seed)
        x, y, z = rng.normal(size=(3, N_samples))
        az, el, _ = cart2sph(x, y, z)
        lons_seed, lats_seed = np.rad2deg(az), np.rad2deg(el)

        labels_seed = assign_closest_label(lons_seed, lats_seed, lons, lats, labels)

        lonlats = torch.from_numpy(np.stack([lons_seed, lats_seed])).T
        labels = torch.from_numpy(labels_seed)

    return lonlats, labels

class CheckerboardDataModule(pl.LightningDataModule):
    def __init__(self, num_samples=5000, batch_size=1000, num_classes = 4, num_support = 200):
        super().__init__()
        self.num_samples = num_samples
        self.batch_size=batch_size
        self.num_support = num_support
        self.num_classes = num_classes

        # mean and std distance between clusters given the number of points
        self.mean_dist, self.std_dist = calc_avg_distances(num_support, unit="deg")

    def setup(self, stage: str):
        self.train_ds = TensorDataset(*get_data(N_samples = self.num_samples,
                                                N_support = self.num_support,
                                                n_classes=self.num_classes,
                                                seed=0))
        self.valid_ds = TensorDataset(*get_data(N_samples = self.num_samples,
                                                N_support = self.num_support,
                                                n_classes=self.num_classes,
                                                seed=1))
        self.evalu_ds = TensorDataset(*get_data(N_samples = self.num_samples,
                                                N_support = self.num_support,
                                                n_classes=self.num_classes,
                                                grid=True))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.evalu_ds, batch_size=self.batch_size, shuffle=False)

if __name__ == '__main__':
    main()
