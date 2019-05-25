import json

from dataset import PAC2019, PAC20192D
from model import Model, VGGBasedModel, VGGBasedModel2D
from model_resnet import ResNet, resnet18

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


# def cosine_rampdown(current, rampdown_length):
#     """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
#     assert 0 <= current <= rampdown_length
#     return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))
#
# def sigmoid_rampup(current, rampup_length):
#     if rampup_length == 0:
#         return 1.0
#     else:
#         current = np.clip(current, 0.0, rampup_length)
#         phase = 1.0 - current / rampup_length
#         return float(np.exp(-5.0 * phase * phase))


with open("config.json") as fid:
    ctx = json.load(fid)

# train_set = PAC2019(ctx, set='train')
# val_set = PAC2019(ctx, set='val')
train_set = PAC20192D(ctx, set='train', split=0.8)
val_set = PAC20192D(ctx, set='val', split=0.8)

train_loader = DataLoader(train_set, shuffle=True, drop_last=True,
                             num_workers=8, batch_size=ctx["batch_size"])
val_loader = DataLoader(val_set, shuffle=False, drop_last=False,
                             num_workers=8, batch_size=ctx["batch_size"])

mse_loss = nn.MSELoss()
# model = Model()
# model = VGGBasedModel()
# model = VGGBasedModel2D()
model = resnet18()
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=ctx["learning_rate"],
                             weight_decay=ctx["weight_decay"])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,5,10], gamma=0.1)
best = np.inf
for e in range(ctx["epochs"]):
    scheduler.step()
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
            print('Training Loss: ', torch.mean(torch.stack(last_50)))
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

    if torch.mean(torch.stack(val_mae_loss)) < best:
        best = torch.mean(torch.stack(val_mae_loss))
        print('model saved')
        torch.save(model.state_dict(), 'models/best_model.pt')

    print('Validation Loss (MSE): ', torch.mean(torch.stack(val_mse_loss)))
    print('Validation Loss (MAE): ', torch.mean(torch.stack(val_mae_loss)))

