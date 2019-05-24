import json

from dataset import PAC2019, PAC20192D
from model import Model, VGGBasedModel, VGGBasedModel2D

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import medicaltorch.transforms as mt_transforms
import torchvision as tv
import torchvision.utils as vutils

import matplotlib.pyplot as plt
from collections import defaultdict


with open("config.json") as fid:
    ctx = json.load(fid)

val_set = PAC2019(ctx, set='val', split=0.8)

val_loader = DataLoader(val_set, shuffle=False, drop_last=False,
                             num_workers=8, batch_size=1)

model = VGGBasedModel2D()
model.cuda()
model.load_state_dict(torch.load('models/best_model.pt'))
model.eval()

portion = 0.8
errors = []
error_per_age = defaultdict(list)
for i, data in enumerate(val_loader):
    input_image = Variable(data["input"]).float().cuda()
    print(input_image.shape)


    slices = []
    start = int((1.-portion)*input_image.shape[1])
    end = int(portion*input_image.shape[1])
    input_image = input_image[0,start:end,:,:]
    for slice_idx in range(input_image.shape[0]):
        slice = input_image[slice_idx,:,:]
        slice = slice.unsqueeze(0)
        slices.append({
            'image': slice,
            'label': data['label']
        })
        # print('Slice: ', slice.shape)

    error = []
    for slice in slices:
        slice['image'] = slice['image'].unsqueeze(0)
        # print(slice['image'].shape)
        output = model(slice['image'])
        print(output[0], slice['label'])
        error.append(np.abs(output[0].item() - slice['label'].item()))
    print(error)
    errors.append(error)

    error_per_age[int(slice['label'].item())].append(np.mean(error))

fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)

print(error_per_age)
sorted_values = []
keys = []
for k in sorted(error_per_age.keys()):
    sorted_values.append(error_per_age[k])
    keys.append(k)

ax.boxplot(sorted_values)
ax.set_xticklabels(keys)
plt.show()


errors = np.array(errors)
print(errors.shape)
mean_errors = np.mean(errors, axis=0)
# plt.plot(mean_errors)
fig, (ax,ax2) = plt.subplots(nrows=2, sharex=True)
x = np.linspace(0, errors.shape[1])
extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
ax.imshow(mean_errors[np.newaxis,:], cmap="viridis", aspect="auto", extent=extent)
# print(mean_errors.shape)
# print(x.shape)
ax2.plot(np.arange(mean_errors.shape[0]),mean_errors)

plt.ylabel('Mean Absolute Error (MAE)')
plt.xlabel('Slice index')

plt.show()
print(mean_errors)
