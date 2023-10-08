import io
import requests
from urllib import request 
from zipfile import ZipFile
from pathlib import Path
import csv
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, IterableDataset, DataLoader
import sklearn.datasets
from functools import reduce

def get_election_data(norm_x=True, norm_y=True, coords_as_feats=False):
    '''
    Download and process the Election dataset used in CorrelationGNN (https://arxiv.org/abs/2002.08274)

    Parameters:
    norm_x = logical; should features be normalized
    norm_y = logical; should outcome be normalized
    coords_as_feats = logical; should lat/lon coordinates be added as features

    Return:
    coords = spatial coordinates (lon/lat)
    x = features at location (excluding outcome variable)
    y = outcome variable
    '''
    path_to_data = './data/election'

    c = torch.load(path_to_data + '/c.pt')
    x = torch.load(path_to_data + '/x.pt')
    y = torch.load(path_to_data + '/y.pt')

    if norm_y==True:
        y = ((y - y.min()) / (y.max() - y.min()))
    if norm_x==True:
        for i in range(x.shape[1]):
            x[:,i] = ((x[:,i] - x[:,i].min()) / (x[:,i].max() - x[:,i].min()))
    if coords_as_feats:
            x = torch.cat((x,c),1)

    return c, x, y

def get_cali_housing_data(norm_x=True, norm_y=True, coords_as_feats=False, add_coord_noise=True):
    '''
    Download and process the California Housing Dataset

    Parameters:
    norm_x = logical; should features be normalized
    norm_y = logical; should outcome be normalized
    coords_as_feats = logical; should lat/lon coordinates be added as features
    add_coord_noise = logical; should a tiny Gaussian noise be added to lat/lon values (to distinguish duplicate coordinates)

    Return:
    coords = spatial coordinates (lon/lat)
    x = features at location
    y = outcome variable
    '''
    path_to_data = './data/cali_housing'

    c = torch.load(path_to_data + '/c.pt')
    x = torch.load(path_to_data + '/x.pt')
    y = torch.load(path_to_data + '/y.pt')

    if add_coord_noise: 
        noise = torch.randn(c.shape) * 1e-7
        c = c + noise
    if norm_y==True:
        y = ((y - y.min()) / (y.max() - y.min()))
    if norm_x==True:
        for i in range(x.shape[1]):
            x[:,i] = ((x[:,i] - x[:,i].min()) / (x[:,i].max() - x[:,i].min()))
    if coords_as_feats:
        x = torch.cat((x,c),1)

    return c, x, y

def get_air_temp_data(norm_y=True, norm_x=True, coords_as_feats=False):
    '''
    Download and process the Global Air Temperature dataset

    Parameters:
    norm_y = logical; should outcome be normalized
    norm_x = logical; should features be normalized
    coords_as_feats = logical; should lat/lon coordinates be added as features

    Return:
    c = spatial coordinates (lon/lat)
    x = features at location
    y = outcome variable
    '''
    path_to_data = './data/air_temp'

    c = torch.load(path_to_data + '/c.pt')
    x = torch.load(path_to_data + '/x.pt')
    y = torch.load(path_to_data + '/y.pt')

    if norm_y==True:
        y = ((y - y.min()) / (y.max() - y.min()))
    if norm_x==True:
        x = ((x - x.min()) / (x.max() - x.min()))
    if coords_as_feats:
        x = torch.cat((x.reshape(-1,1),c),1)

    return c, x.reshape(x.shape[0],-1), y