from torch import nn
import torch
import numpy as np
import functools


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def get_random_lonlats(batch_size):
    """
    samples random points on the sphere returns [N x 2] tensor of lon, lat coordinates in degree
    """
    x, y, z = torch.normal(mean=torch.zeros((3, batch_size)),std=torch.tensor(1.))
    az, el, _ = cart2sph(x, y, z)
    lons, lats = torch.rad2deg(az), torch.rad2deg(el)
    return torch.stack([lons, lats]).T

def log_loss(pred):
    return -torch.log(pred + 1e-5)

def get_loss_fn(presence_only=False, loss_weight=2048, regression=False):
    """
    generate a loss function
    :param presence_only: if True, calculate presence-only loss as in Cole and al., 2023
    :param loss_weight: weight used in the presence-only loss function
    :return: a loss function loss_fn(model, lonlats, label)
    """

    if regression:
        return MSE_loss
        
    elif presence_only:
        return functools.partial(full_loss, loss_weight=loss_weight)

    else: # normal presence absence loss
        return AN_loss

def MSE_loss(model, lonlats, label):
    """MSE on logits."""
    prediction_logits = model.forward(lonlats)

    if prediction_logits.size(1) == 1:
        return nn.functional.mse_loss(prediction_logits, label, reduction='mean')
    else:
        return nn.functional.mse_loss(prediction_logits, label.squeeze(), reduction='mean') 
        
def SLDS_loss(model, lonlats, label):
    """The "assume negative" loss (same location, different species) from Cole and al., 2023."""
        
    prediction_logits = model.forward(lonlats)
    
    # loss at data location
    batch_size, N_classes = prediction_logits.shape
    pos_logits = prediction_logits[torch.arange(batch_size, device=label.device).unsqueeze(1), label]
    neg_logits = prediction_logits[torch.arange(batch_size, device=label.device).unsqueeze(1), 
                                   torch.randint(N_classes, size=(batch_size, 1), device=label.device)]
    loss_dl = log_loss(torch.sigmoid(pos_logits)).sum() + log_loss(1 - torch.sigmoid(neg_logits)).sum()
    
    return loss_dl/batch_size

def SSDL_loss(model, lonlats, label):
    """The "assume negative" loss (same species, different location) from Cole and al., 2023."""
        
    batch_size = lonlats.size(0)
    lonlats_negatives = get_random_lonlats(batch_size).to(lonlats.device).to(lonlats.dtype)

    # stack data locations and random locations to run them all together through the model
    lonlats_stacked = torch.cat((lonlats, lonlats_negatives), 0)
    logits_stacked = model.forward(lonlats_stacked)

    # split again in data locations and random locations
    prediction_logits = logits_stacked[:batch_size]
    loc_pred_rand = logits_stacked[batch_size:]
    
    # loss at data location
    pos_logits = prediction_logits[torch.arange(batch_size, device=label.device).unsqueeze(1), label]
    loss_dl = log_loss(torch.sigmoid(pos_logits)).mean()
    
    # loss at random location
    N_classes = prediction_logits.size(1)
    neg_logits = loc_pred_rand[torch.arange(batch_size, device=label.device).unsqueeze(1), 
                               torch.randint(N_classes, size=(batch_size, 1), device=label.device)]
    loss_rl = log_loss(1 - torch.sigmoid(neg_logits)).mean()
    
    return loss_dl + loss_rl

def full_loss(model, lonlats, label, loss_weight):
    """The full "assume negative" loss from Cole and al., 2023, combining SSDL and SLDS.
        Similar to the Geoprior loss of Mac Aodha et al., 2019, but without the photographer losses."""
    
    batch_size = lonlats.size(0)
    lonlats_negatives = get_random_lonlats(batch_size).to(lonlats.device).to(lonlats.dtype)

    # stack data locations and random locations to run them all together through the model
    lonlats_stacked = torch.cat((lonlats, lonlats_negatives), 0)
    logits_stacked = model.forward(lonlats_stacked)

    # split again in data locations and random locations
    prediction_logits = logits_stacked[:batch_size]
    loc_pred_rand = logits_stacked[batch_size:]
    
    # loss at data location
    N_classes = prediction_logits.size(1)
    pos_logits = prediction_logits[torch.arange(batch_size, device=label.device).unsqueeze(1), label]
    neg_logits = prediction_logits[torch.arange(N_classes, device=label.device) != label] 
    loss_dl_pos = log_loss(torch.sigmoid(pos_logits)).sum() / (batch_size * N_classes)
    loss_dl_neg = log_loss(1 - torch.sigmoid(neg_logits)).sum() / (batch_size * N_classes)
    
    # loss at random location
    loss_rl = log_loss(1 - torch.sigmoid(loc_pred_rand)).sum(axis=1).mean() / N_classes
    
    return loss_weight * loss_dl_pos + loss_dl_neg + loss_rl

def AN_loss(model, lonlats, label):
    """The simple "assume negative" loss from Cole and al., 2021. This is the standard cross-entropy loss."""
    prediction_logits = model.forward(lonlats)
    if prediction_logits.size(1) == 1:
        return nn.functional.binary_cross_entropy_with_logits(prediction_logits, label)
    else:
        return nn.functional.cross_entropy(prediction_logits, label.squeeze())
