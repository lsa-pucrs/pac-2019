import os
import numpy as np
import pandas as pd
import nibabel as nib
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import medicaltorch.transforms as mt_transforms
import torchvision as tv
import torchvision.utils as vutils


class PAC2019(Dataset):
    def __init__(self, ctx, set, split=0.8):
        self.ctx = ctx
        dataset_path = ctx["dataset_path"]
        csv_path = os.path.join(dataset_path, "PAC2019_BrainAge_Training.csv")
        dataset = []
        stratified_dataset = []

        with open(csv_path) as fid:
            for i, line in enumerate(fid):
                if i == 0:
                    continue
                line = line.split(',')
                dataset.append({
                    'subject': line[0],
                    'age': float(line[1]),
                    'gender': line[2],
                    'site': int(line[3].replace('\n',''))
                })

        sites = defaultdict(list)
        for data in dataset:
            sites[data['site']].append(data)

        for site in sites.keys():
            length = len(sites[site])
            if set == 'train':
                stratified_dataset += sites[site][0:int(length*split)]
            if set == 'val':
                stratified_dataset += sites[site][int(length*split):]

        self.dataset = stratified_dataset

        self.transform = tv.transforms.Compose([
            mt_transforms.ToPIL(labeled=False),
            mt_transforms.ElasticTransform(alpha_range=(28.0, 30.0),
                                           sigma_range=(3.5, 4.0),
                                           p=0.3, labeled=False),
            mt_transforms.RandomAffine(degrees=4.6,
                                       scale=(0.98, 1.02),
                                       translate=(0.03, 0.03),
                                       labeled=False),
            mt_transforms.RandomTensorChannelShift((-0.10, 0.10)),
            mt_transforms.ToTensor(labeled=False),
        ])

    def __getitem__(self, idx):
        data = self.dataset[idx]
        filename = os.path.join(self.ctx["dataset_path"], 'gm', data['subject'] + '_gm.nii.gz')
        input_image = torch.FloatTensor(nib.load(filename).get_fdata())
        input_image = input_image.permute(2, 0, 1)

        # transformed = {
        #     'input': input_image
        # }
        # self.transform(transformed)

        # plt.imshow(input_image[60,:,:])
        # plt.show()
        # plt.imshow(input_image[:,60,:])
        # plt.show()
        # plt.imshow(input_image[:,:,60])
        # plt.show()
        #
        # raise


        return {
            'input': input_image,
            'label': data['age']
        }

    def __len__(self):
        return len(self.dataset)


class PAC20192D(Dataset):
    def __init__(self, ctx, set, split=0.7, portion=0.8):
        """
        split: train/val split
        portion: portion of the axial slices that enter the dataset
        """
        self.ctx = ctx
        self.portion = portion
        dataset_path = ctx["dataset_path"]
        csv_path = os.path.join(dataset_path, "PAC2019_BrainAge_Training.csv")
        dataset = []
        stratified_dataset = []

        with open(csv_path) as fid:
            for i, line in enumerate(fid):
                if i == 0:
                    continue
                line = line.split(',')
                dataset.append({
                    'subject': line[0],
                    'age': float(line[1]),
                    'gender': line[2],
                    'site': int(line[3].replace('\n',''))
                })

        sites = defaultdict(list)
        for data in dataset:
            sites[data['site']].append(data)

        for site in sites.keys():
            length = len(sites[site])
            if set == 'train':
                stratified_dataset += sites[site][0:int(length*split)]
            if set == 'val':
                stratified_dataset += sites[site][int(length*split):]


        self.dataset = stratified_dataset
        self.slices = []

        self.transform = tv.transforms.Compose([
            mt_transforms.ToPIL(labeled=False),
            mt_transforms.ElasticTransform(alpha_range=(28.0, 30.0),
                                           sigma_range=(3.5, 4.0),
                                           p=0.3, labeled=False),
            mt_transforms.RandomAffine(degrees=4.6,
                                       scale=(0.98, 1.02),
                                       translate=(0.03, 0.03),
                                       labeled=False),
            mt_transforms.RandomTensorChannelShift((-0.10, 0.10)),
            mt_transforms.ToTensor(labeled=False),
        ])

        self.preprocess_dataset()



    def preprocess_dataset(self):
        for i, data in enumerate(self.dataset):
            if i % 50 == 0:
                print('Loading %d/%d' % (i, len(self.dataset)))
            filename = os.path.join(self.ctx["dataset_path"], 'gm', data['subject'] + '_gm.nii.gz')
            input_image = torch.FloatTensor(nib.load(filename).get_fdata())
            input_image = input_image.permute(2, 0, 1)

            start = int((1.-self.portion)*input_image.shape[0])
            end = int(self.portion*input_image.shape[0])
            input_image = input_image[start:end,:,:]
            for slice_idx in range(input_image.shape[0]):
                slice = input_image[slice_idx,:,:]
                slice = slice.unsqueeze(0)
                self.slices.append({
                    'image': slice,
                    'age': data['age']
                })



    def __getitem__(self, idx):
        # data = self.dataset[idx]
        # filename = os.path.join(self.ctx["dataset_path"], 'gm', data['subject'] + '_gm.nii.gz')
        # input_image = torch.FloatTensor(nib.load(filename).get_fdata())
        # input_image = input_image.permute(2, 0, 1)

        data = self.slices[idx]
        transformed = {
            'input': data['image']
        }
        transformed = self.transform(transformed)

        return {
            'input': transformed['input'],
            'label': data['age']
        }

    def __len__(self):
        return len(self.slices)
