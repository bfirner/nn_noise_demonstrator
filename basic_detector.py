#! /usr/bin/python3

import argparse
import numpy
import os
import random
import shutil
import torch

from PIL import Image

# Mostly the example network from pytorch docs
class Net(torch.nn.Module):

    def __init__(self, input_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.conv_out_size = input_size
        for _ in range(3):
            self.conv_out_size = (self.conv_out_size - 5 + 2 * 2) // 2 + 1
        for _ in range(2):
            self.conv_out_size = (self.conv_out_size - 3 + 2 * 1) // 2 + 1
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(6),
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * self.conv_out_size * self.conv_out_size, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1))

    def forward(self, x):
        return self.net.forward(x)

def getFIRFilterLayer(kernel_size):
    # Only support 3 in spite of the argument! Muahaha!
    assert kernel_size == 3
    conv_blur = torch.nn.Conv2d(in_channels=1, out_channels=1,
            kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
    with torch.no_grad():
        # Need to find the total tiles
        base = 1./16.
        conv_blur.weight.copy_(torch.tensor(
            [[base, 2 * base, base],
             [2 * base, 4 * base, 2 * base],
             [base, 2 * base, base]]))
    return conv_blur.cuda()


def getLineData(input_size, noise, blur, intensity, batch_size=32):
    # Noise with variance 1/100
    batch = torch.zeros((batch_size, 1, input_size, input_size))
    labels = torch.zeros(batch_size, 1)
    # Randomly put a "target" in some of the batches and set the label to 1
    for i in range(batch_size):
        if 1 == random.randint(0, 1):
            # The target is a vertical line
            labels[i] = 1.
            # Find an x location
            x = random.randint(1, batch.size(3) - 1)
            y = random.randint(0, batch.size(2) - 10)
            batch[i, :, y:y+10, x] = intensity
    # Add noise
    batch += torch.randn((batch_size, 1, input_size, input_size)).abs() * noise
    batch = batch.cuda()
    labels = labels.cuda()
    # Add blur
    if blur is not None:
        with torch.no_grad():
            batch = blur(batch)
    return (batch, labels)

inparser = argparse.ArgumentParser(
    description="Arguments for the noise training script.")
inparser.add_argument(
    '--noise', type=float, default=0.1,
    help='Noise will be the absolute value of a normal with this standard deviation.')
inparser.add_argument(
    '--intensity', type=float, default=0.3,
    help='The intensity of the signal (in the range 0 to 1).')
inparser.add_argument(
    '--size', type=int, default=250,
    help='The size of the training images.')
inparser.add_argument(
    '--shape', type=str, default="vertical_line",
    help='The shape of the target (options: vertical_line).')
args = inparser.parse_args()

input_size = args.size
noise = args.noise
shape = args.shape
intensity = args.intensity
blur = getFIRFilterLayer(3)
net = Net(input_size = input_size).cuda()
optimizer = torch.optim.Adam(net.parameters())
loss_fn = torch.nn.L1Loss()

for batch_num in range(2000):
    if "vertical_line" == shape:
        batch, labels = getLineData(input_size, noise, blur, intensity, batch_size=32)
    else:
        raise RuntimeError(f"Shape {shape} is not supported.")
    optimizer.zero_grad()
    out = net.forward(batch)
    loss = loss_fn(out, labels)
    if 0 == batch_num % 100:
        with torch.no_grad():
            print(f"Batch {batch_num} loss is {loss.mean()}")
    loss.backward()
    optimizer.step()

# TODO FIXME Also lower the signal strength so that it moves closer to the noise floor
outpath = f'{shape}_{intensity}_intensity_{noise}_noise_blur'
# Remove previous results if they exist
if os.path.isdir(outpath):
    shutil.rmtree(outpath)
os.mkdir(outpath)
net.eval()
batch, labels = getLineData(input_size, noise, blur, intensity, batch_size=1000)
out = net.forward(batch.cuda())
prediction = (out > 0.5).to(torch.uint8)
correct = prediction == labels
for img_num in range(batch.size(0)):
    pixels = ((batch[img_num, 0]) * 255).to(torch.uint8).to('cpu').numpy()
    img = Image.frombytes(
        mode="L", size=(input_size, input_size), data=pixels)
    img.save(fp=f'{outpath}/{img_num}.png')
with open(f"{outpath}/status.csv", "w") as f:
    # File in the status of each sample (label and prediction)
    for example in range(prediction.size(0)):
        f.write(f"{example}, {labels[example].to(torch.uint8).item()}, {prediction[example].item()}\n")
    f.write(f"Mean, 1.0, {(correct.sum()/1000.).item()}\n")
