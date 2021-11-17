import os
import glob
import numpy as np
from matplotlib import pyplot
from PIL import Image
from zipfile import ZipFile
import torch
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import params
import matplotlib.pyplot as plt
import seaborn as sns
import math
import cv2
import pandas as pd

def get_item(img_path):

    transform = transforms.Compose([
                            transforms.Resize(112),
                            transforms.ToTensor()
                            ])

    torch_img = Image.open(img_path)
    torch_img = transform(torch_img)
    return torch_img

def load_stored_data(path):
    data = np.load(path)
    data = torch.from_numpy(data)
    data_mean = torch.mean(data, 0, True)
    data_median, _ = torch.median(data, 0, True)
    data_std_mad = torch.std((data-data_median)**2, dim=0, keepdim=True)
    data_std = torch.std(data, dim=0, keepdim=True)
    return data_mean, data_std, data_median, data_std_mad

def get_heatmap(cam):
    cam = F.interpolate(cam, size=(112, 112), mode='bilinear', align_corners=False)
    cam = 255 * cam.squeeze()
    # this step changes the value range to [0, 255]
    heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1))
    heatmap = heatmap.float() / 255
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    return  heatmap

def plot_qos_shift(data_high, data_low, method=None, y_title=None):

    sns.kdeplot(data_high, color='r', shade=True, Label='high')
    sns.kdeplot(data_low, color='b', shade=True, Label='low')
    plt.xlabel(y_title, fontsize=18)
    plt.ylabel('PDF', fontsize=18)
    plt.legend()
    plt.tight_layout()
    # plt.savefig("./samples/results/{}_shift.jpg".format(method), bbox_inches='tight')
    # plt.close()
    plt.show()

def plot_2d_heatmap(heatmap, title=None, append_score=False, score=None, score_2=None):
    # heatmap = .5 * get_heatmap(torch.unsqueeze(data_test, 0))  + 1. * img_org
    # heatmap = get_heatmap(torch.unsqueeze(heatmap, 0))
    if append_score:
        size = 30, 112, 3
        m = np.zeros(size, dtype=np.uint8)
        m = cv2.putText(m, '{:2,.2f} | {:2,.2f}'.format(score, score_2), (1, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA)
        heatmap = np.clip(heatmap.permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
        heatmap = np.concatenate((heatmap, m), axis=0)
    else:
        heatmap = np.clip(heatmap.permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)

    # cv2.imwrite("{}.jpg".format(title), cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    cv2.imshow("{}".format(title), cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)

def main():

    methods = ["BRISQUE", "MagFace", "FaceQnet", "Serfiq"]

    for method in methods:
        data_high_mean, data_high_std, data_high_median, data_high_std_median = load_stored_data(path="./activation_weights/{}_high.npy".format(method))
        print("High: ", torch.max(data_high_std), torch.min(data_high_std))

        # Mean average mapping for high
        plot_2d_heatmap(get_heatmap(torch.unsqueeze(data_high_mean, 0)) , title="{}_high_mean".format(method))
        plot_2d_heatmap(get_heatmap(torch.unsqueeze(data_high_std.div(data_high_std.max()), 0)), title="{}_high_std".format(method))

        # Median average mapping
        plot_2d_heatmap(get_heatmap(torch.unsqueeze(data_high_median, 0)), title="{}_high_median".format(method))
        plot_2d_heatmap(get_heatmap(torch.unsqueeze(data_high_std_median.div(data_high_std_median.max()), 0)),
                        title="{}_high_std_median".format(method))

        data_low_mean, data_low_std, data_low_median, data_low_std_median = load_stored_data(path="./activation_weights/{}_low.npy".format(method))
        print("Low: ", torch.max(data_low_std), torch.min(data_low_std))

        # Mean average mapping for low
        plot_2d_heatmap(get_heatmap(torch.unsqueeze(data_low_mean, 0)), title="{}_low_median".format(method))
        plot_2d_heatmap(get_heatmap(torch.unsqueeze(data_low_std.div(data_low_std.max()), 0)), title="{}_low_std".format(method))

        # Median average mapping for low
        plot_2d_heatmap(get_heatmap(torch.unsqueeze(data_low_median, 0)), title="{}_low_median".format(method))
        plot_2d_heatmap(get_heatmap(torch.unsqueeze(data_low_std_median.div(data_low_std_median.max()), 0)),
                        title="{}_low_std".format(method))

        plot_qos_shift(torch.flatten(data_high_std), torch.flatten(data_low_std), method="{}_std".format(method), y_title="std_median")
        plot_qos_shift(torch.flatten(data_high_median), torch.flatten(data_low_median), method="{}_mean".format(method), y_title="median")

        # variance with mean difference
        diff = torch.abs(data_high_std - data_low_std)
        print("Diff: ", torch.max(diff), torch.min(diff))
        plot_2d_heatmap(get_heatmap(torch.unsqueeze(diff.div(diff.max()), 0)), title="{}_diff_std".format(method))

        # deviation with median difference
        diff = torch.abs(data_high_std_median - data_low_std_median)
        plot_2d_heatmap(get_heatmap(torch.unsqueeze(diff.div(diff.max()), 0)), title="{}_diff_std_median".format(method))

        # overlap with groundtruth image
        img_org = get_item(
            "data/n004999/0015_01.jpg")
        # single image activation map
        hmp = 0.5 * get_heatmap(torch.unsqueeze(data_high_mean, 0)) + 1.0 * img_org
        plot_2d_heatmap(hmp, title="{}_real_image".format(method))

if __name__ == '__main__':
    main()
