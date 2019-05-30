import json

from dataset import PAC2019, PAC20192D, PAC20193D
from model import Model, VGGBasedModel, VGGBasedModel2D, ColeModel
from model_resnet import ResNet, resnet18

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from tqdm import *

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def cosine_lr(current_epoch, num_epochs, initial_lr):
    return initial_lr * cosine_rampdown(current_epoch, num_epochs)

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


with open("config.json") as fid:
    ctx = json.load(fid)

if ctx["3d"]:
    train_set = PAC20193D(ctx, set='train')
    val_set = PAC20193D(ctx, set='val')
    model = ColeModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=ctx["learning_rate"],
                                 momentum=0.9, weight_decay=ctx["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)

else:
    train_set = PAC20192D(ctx, set='train', split=0.8)
    val_set = PAC20192D(ctx, set='val', split=0.8)
    model = resnet18()
    optimizer = torch.optim.Adam(model.parameters(), lr=ctx["learning_rate"],
                                 weight_decay=ctx["weight_decay"])

train_loader = DataLoader(train_set, shuffle=True, drop_last=True,
                             num_workers=8, batch_size=ctx["batch_size"])
val_loader = DataLoader(val_set, shuffle=False, drop_last=False,
                             num_workers=8, batch_size=ctx["batch_size"])

mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
model.cuda()

best = np.inf
for e in tqdm(range(1, ctx["epochs"]+1), desc="Epochs"):
    model.train()
    last_50 = []

    if ctx["3d"]:
        scheduler.step()
        tqdm.write('Learning Rate: {:.6f}'.format(scheduler.get_lr()[0]))
    else:
        if e <= ctx["initial_lr_rampup"]:
            lr = ctx["learning_rate"] * sigmoid_rampup(e, ctx["initial_lr_rampup"])
        else:
            lr = cosine_lr(e-ctx["initial_lr_rampup"],
                             ctx["epochs"]-ctx["initial_lr_rampup"],
                             ctx["learning_rate"])

        for param_group in optimizer.param_groups:
            tqdm.write("Learning Rate: {:.6f}".format(lr))
            param_group['lr'] = lr


    for i, data in enumerate(train_loader):
        if ctx["mixup"]:
            lam = np.random.beta(ctx["mixup_alpha"], ctx["mixup_alpha"])

            length_data = data["input"].size(0)//2
            data1_x = data["input"][0:length_data]
            data1_y = data["label"][0:length_data]
            data2_x = data["input"][length_data:]
            data2_y = data["label"][length_data:]

            data["input"] = lam*data1_x + (1.-lam)*data2_x
            data["label"] = lam*data1_y + (1.-lam)*data2_y

        # print(data["input"].shape)
        input_image = Variable(data["input"], requires_grad=True).float().cuda()
        output = model(input_image)
        label = Variable(data["label"].float()).cuda()
        # print(output)
        # print(label)

        loss = mae_loss(output.squeeze(), label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        last_50.append(loss.data)
        if (i+1) % 50 == 0:
            tqdm.write('Training Loss: %f' % torch.mean(torch.stack(last_50)).item())
            last_50 = []


    # tqdm.write('Validation...')
    model.eval()
    # val_mse_loss = []
    val_mae_loss = []
    for i, data in enumerate(val_loader):
        input_image = Variable(data["input"]).float().cuda()
        output = model(input_image)
        label = Variable(data["label"].float()).cuda()

        loss = mae_loss(output.squeeze(), label)
        val_mae_loss.append(loss.data)


        # loss = torch.mean(torch.abs(output.squeeze() - label))
        # val_mae_loss.append(loss.data)

    if torch.mean(torch.stack(val_mae_loss)) < best:
        best = torch.mean(torch.stack(val_mae_loss))
        tqdm.write('model saved')
        torch.save(model.state_dict(), ctx["save_path"])

    # print('Validation Loss (MSE): ', torch.mean(torch.stack(val_mse_loss)))
    tqdm.write('Validation Loss (MAE): %f' % torch.mean(torch.stack(val_mae_loss)).item())
