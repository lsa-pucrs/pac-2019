import json

from dataset import PAC2019
from model import Model, VGGBasedModel

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


with open("config.json") as fid:
    ctx = json.load(fid)
train_set = PAC2019(ctx, set='train')
val_set = PAC2019(ctx, set='val')
train_loader = DataLoader(train_set, shuffle=True, drop_last=True,
                             num_workers=8, batch_size=ctx["batch_size"])
val_loader = DataLoader(val_set, shuffle=False, drop_last=False,
                             num_workers=8, batch_size=ctx["batch_size"])

mse_loss = nn.MSELoss()
# model = Model()
model = VGGBasedModel()
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=ctx["learning_rate"],
                             weight_decay=ctx["weight_decay"])

for e in range(ctx["epochs"]):
    model.train()
    last_50 = []
    for i, data in enumerate(train_loader):
        # print(data["input"].shape)
        input_image = Variable(data["input"], requires_grad=True).float().cuda()
        output = model(input_image)
        label = Variable(data["label"].float()).cuda()
        # print(output)
        # print(label)

        loss = mse_loss(output.squeeze(), label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        last_50.append(loss.data)
        if (i+1) % 50 == 0:
            print('Training Loss: ', np.mean(last_50))
            last_50 = []


    print('Validation...')
    model.eval()
    val_mse_loss = []
    val_mae_loss = []
    for i, data in enumerate(val_loader):
        input_image = Variable(data["input"]).float().cuda()
        output = model(input_image)
        label = Variable(data["label"].float()).cuda()

        loss = mse_loss(output.squeeze(), label)
        val_mse_loss.append(loss.data)

        loss = torch.mean(torch.abs(output.squeeze() - label))
        val_mae_loss.append(loss.data)

    print('Validation Loss (MSE): ', np.mean(val_mse_loss))
    print('Validation Loss (MAE): ', np.mean(val_mae_loss))
