import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import json
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader
import lightning as pl
import pandas as pd
import torch

VAL_SPLIT_RATIO = 0.8

# some categories of interest
ARCTIC_FOX = "Vulpes lagopus" # fig 1 CSP paper
RED_FOX = "Vulpes vulpes"
KIT_FOX = "Vulpes macrotis"
BAT_EARED_FOX = "Otocyon megalotis"  # fig 1 CSP paper
WOOD_THRUSH = "Hylocichla mustelina" # GeoPrior Fig 1
MONARCH_BUTTERFLY = "Danaus plexippus"
AFRICAN_BUSH_ELEPHANT = "Loxodonta africana"
WESTERN_HONEY_BEE = "Apis mellifera"

QUALITATIVE_SPECIES = [ARCTIC_FOX, RED_FOX, BAT_EARED_FOX, WOOD_THRUSH, MONARCH_BUTTERFLY, AFRICAN_BUSH_ELEPHANT, WESTERN_HONEY_BEE]
QUALITATIVE_SPECIES_NAMES = ["Arctic Fox", "Red Fox", "Bat Eared Fox", "Wood Thrush", "Monarch Butterfly", "African Bush Elephant", "Western Honey Bee"]


def default_loader(path):
    return Image.open(path).convert('RGB')

def load_taxonomy(ann_data, tax_levels, classes):
    # loads the taxonomy data and converts to ints
    taxonomy = {}

    if 'categories' in ann_data.keys():
        num_classes = len(ann_data['categories'])
        for tt in tax_levels:
            tax_data = [aa[tt] for aa in ann_data['categories']]
            _, tax_id = np.unique(tax_data, return_inverse=True)
            taxonomy[tt] = dict(zip(range(num_classes), list(tax_id)))
    else:
        # set up dummy data
        for tt in tax_levels:
            taxonomy[tt] = dict(zip([0], [0]))

    # create a dictionary of lists containing taxonomic labels
    classes_taxonomic = {}
    for cc in np.unique(classes):
        tax_ids = [0]*len(tax_levels)
        for ii, tt in enumerate(tax_levels):
            tax_ids[ii] = taxonomy[tt][cc]
        classes_taxonomic[cc] = tax_ids

    return taxonomy, classes_taxonomic

class INAT_ImageDataset(data.Dataset):
    def __init__(self, root, ann_file, is_train=True, return_location=False, logits_file=None):

        # load annotations
        print('Loading annotations from: ' + os.path.basename(ann_file))
        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        # set up the filenames and annotations
        self.imgs = [aa['file_name'] for aa in ann_data['images']]
        self.ids = [aa['id'] for aa in ann_data['images']]

        # if we dont have class labels set them to '0'
        if 'annotations' in ann_data.keys():
            self.classes = [aa['category_id'] for aa in ann_data['annotations']]
        else:
            self.classes = [0]*len(self.imgs)

        # load taxonomy
        self.tax_levels = ['id', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']
                           #8142, 4412,    1120,     273,     57,      25,       6
        self.taxonomy, self.classes_taxonomic = load_taxonomy(ann_data, self.tax_levels, self.classes)

        # print out some stats
        print('\t' + str(len(self.imgs)) + ' images')
        print('\t' + str(len(set(self.classes))) + ' classes')

        self.root = root
        self.is_train = is_train
        self.loader = default_loader

        # augmentation params
        self.im_size = [299, 299]  # can change this to train on higher res
        self.mu_data = [0.485, 0.456, 0.406]
        self.std_data = [0.229, 0.224, 0.225]
        self.brightness = 0.4
        self.contrast = 0.4
        self.saturation = 0.4
        self.hue = 0.25

        # augmentations
        self.center_crop = transforms.CenterCrop((self.im_size[0], self.im_size[1]))
        self.scale_aug = transforms.RandomResizedCrop(size=self.im_size[0])
        self.flip_aug = transforms.RandomHorizontalFlip()
        self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        self.tensor_aug = transforms.ToTensor()
        self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)

        # load logits
        self.logits_file = logits_file
        if self.logits_file is not None:
            self.logits = np.load(self.logits_file, mmap_mode='r+')
            
        self.return_location = return_location
        if return_location:
            # location data
            path, ext = os.path.splitext(ann_file)
            locations, classes, users, dates, keep_indxs, data_ids = load_inat_location_data(root, os.path.basename(path) + "_locations" + ext, os.path.basename(ann_file))
            print(f"dropping {len(self.ids) - len(data_ids)} of {len(self.ids)} samples due to lacking location")

            # this can be sped up with some np array conversion instead of looping over a list...
            keep_mask = np.in1d(self.ids, data_ids)

            self.locations = locations
            self.ids = np.array(self.ids)[keep_mask]
            self.classes = np.array(self.classes)[keep_mask]
            self.imgs = np.array(self.imgs)[keep_mask]

    def __getitem__(self, index):
        im_id = self.ids[index]
        species_id = self.classes[index]
        tax_ids = self.classes_taxonomic[species_id]
        
        if self.logits_file is None:
            path = self.root + self.imgs[index]
            img = self.loader(path)
            if self.is_train:
                img = self.scale_aug(img)
                img = self.flip_aug(img)
                img = self.color_aug(img)
            else:
                img = self.center_crop(img)
            img = self.tensor_aug(img)
            img = self.norm_aug(img)

            item = img, im_id, species_id, tax_ids
            
        else:
            logit = self.logits[index]
            logit = torch.from_numpy(logit).float()
            
            # Yield the logits instead of the image
            item = logit, im_id, species_id, tax_ids
        
        if self.return_location:
            item = item + (self.locations[index],)
            
        return item

    def __len__(self):
        return len(self.imgs)
    
    def compute_logits(self, model, logits_file):
        """Compute logits for all images in the dataset using the provided model
        and store them in logits_file."""
        
        batch_size = 11
        model.eval()
        dataloader = iter(data.DataLoader(self, batch_size=batch_size, shuffle=False))
        logits = np.zeros((len(self.ids), 8142))
        i = 0
        while True:
            i += 1
            try:
                batch = next(dataloader)
                indices = np.in1d(self.ids, batch[1].detach().cpu().numpy()).nonzero()[0]
                logits[indices] = model(batch[0].cuda()).detach().cpu().numpy().tolist()
                if i * batch_size % 1001 == 0:
                    print(f"{i * batch_size} images processed")
            except StopIteration:
                break
        
        np.save(logits_file, logits)

    def compute_features(self, model, logits_file):
        """Compute logits for all images in the dataset using the provided model
        and store them in logits_file."""

        batch_size = 11
        model.eval()
        dataloader = iter(data.DataLoader(self, batch_size=batch_size, shuffle=False))
        logits = np.zeros((len(self.ids), 8142))
        i = 0
        while True:
            i += 1
            try:
                batch = next(dataloader)
                indices = np.in1d(self.ids, batch[1].detach().cpu().numpy()).nonzero()[0]
                logits[indices] = model(batch[0].cuda()).detach().cpu().numpy().tolist()
                if i * batch_size % 1001 == 0:
                    print(f"{i * batch_size} images processed")
            except StopIteration:
                break

        np.save(logits_file, logits)

def load_inat_location_data(ip_dir, loc_file_name, ann_file_name, remove_empty=True):
    print('Loading ' + os.path.basename(loc_file_name))

    # load location info
    with open(ip_dir + loc_file_name) as da:
        loc_data = json.load(da)
    loc_data_dict = dict(zip([ll['id'] for ll in loc_data], loc_data))

    if '_large' in loc_file_name:
        # special case where the loc data also includes meta data such as class
        locs = [[ll['lon'], ll['lat']] for ll in loc_data]
        dates = [ll['date_c'] for ll in loc_data]
        classes = [ll['class'] for ll in loc_data]
        users = [ll['user_id'] for ll in loc_data]
        keep_inds = np.arange(len(locs))
        print('\t {} valid entries'.format(len(locs)))

    else:
        # otherwise load regualar iNat data

        # load annotation info
        with open(ip_dir + ann_file_name) as da:
            data = json.load(da)

        ids = [tt['id'] for tt in data['images']]
        ids_all = [ii['image_id'] for ii in data['annotations']]
        classes_all = [ii['category_id'] for ii in data['annotations']]
        classes_mapping = dict(zip(ids_all, classes_all))

        # store locations and associated classes
        locs    = []
        classes = []
        users   = []
        dates   = []
        miss_cnt = 0
        keep_inds = []
        dataids = []
        for ii, tt in enumerate(ids):

            if remove_empty and ((loc_data_dict[tt]['lon'] is None) or (loc_data_dict[tt]['user_id'] is None)):
                miss_cnt += 1
            else:
                if (loc_data_dict[tt]['lon'] is None):
                    loc = [np.nan, np.nan]
                else:
                    loc = [loc_data_dict[tt]['lon'], loc_data_dict[tt]['lat']]

                if (loc_data_dict[tt]['user_id'] is None):
                    u_id = -1
                else:
                    u_id = loc_data_dict[tt]['user_id']

                locs.append(loc)
                classes.append(classes_mapping[int(tt)])
                users.append(u_id)
                dates.append(loc_data_dict[tt]['date_c'])
                keep_inds.append(ii)
                dataids.append(tt)

        print('\t {} valid entries'.format(len(locs)))
        if remove_empty:
            print('\t {} entries excluded with missing meta data'.format(miss_cnt))

    return torch.tensor(locs), torch.tensor(classes).unsqueeze(-1), \
           torch.tensor(users), torch.tensor(dates), torch.tensor(keep_inds), torch.tensor(dataids)

class Inat2018DataModule(pl.LightningDataModule):
    def __init__(self, root, batch_size=2000, mode="location", num_workers=0):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.mode = mode
        self.num_workers = num_workers

        with open(os.path.join(root, "categories.json")) as da:
            data = json.load(da)
        df = pd.DataFrame(data)[["name", "id"]]

        self.name2id = df.set_index("name").to_dict()["id"]
        self.id2name = df.set_index("id").to_dict()["name"]


    def setup(self, stage=None):
        if self.mode == "image" or self.mode == "all":
            return_location = self.mode == "all"
            image_dataset = INAT_ImageDataset(self.root, os.path.join(self.root, "train2018.json"),
                                 is_train=True, return_location=return_location)
            self.train_ds, self.valid_ds = data.random_split(image_dataset, 
                                                             [VAL_SPLIT_RATIO, 1 - VAL_SPLIT_RATIO])
            self.test_ds = INAT_ImageDataset(self.root, os.path.join(self.root, "val2018.json"),
                                 is_train=False, return_location=return_location, 
                                 logits_file=os.path.join(self.root, "val_logits.npy"))
        elif self.mode == "location":
            locations, classes, *_ = load_inat_location_data(self.root, "train2018_locations.json", "train2018.json")
            self.train_ds, self.valid_ds = data.random_split(TensorDataset(locations, classes), 
                                                             [VAL_SPLIT_RATIO, 1 - VAL_SPLIT_RATIO])
            locations, classes, *_ = load_inat_location_data(self.root, "val2018_locations.json", "val2018.json")
            self.test_ds = TensorDataset(locations, classes)
            
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
