from unet import UNet

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary

import numpy as np
import random
import sys
import os
import cv2
import matplotlib.pyplot as plt # plots
from matplotlib.image import imsave
import PIL
import time
import math



class Dataset(object):
    def __init__(self, x, img_transform=None):
        self.len = len(x)
        self.x_data = x
        self.img_transform = img_transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        img = self.x_data[index]

        seed = random.randrange(sys.maxsize)

        if self.img_transform:
            random.seed(seed)  # apply this seed to img transforms
            img_new = self.img_transform(img)


        return img_new


#helper function for pretty printing of current time and remaining time
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent+.00001)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# function for loading images
def load_train_imgs(path_IMAGE_LIB):
    train_images_ct = [x for x in sorted(os.listdir(path_IMAGE_LIB)) if x[-4:] == '.png']  # Read all the images
    names = []

    IMG_HEIGHT, IMG_WIDTH = 512, 512

    x = np.empty((len(train_images_ct), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
    for i, name in enumerate(train_images_ct):
        im = cv2.imread(path_IMAGE_LIB + name, cv2.IMREAD_GRAYSCALE).astype('int16').astype('float32')
        im = (im - np.min(im)) / (np.max(im) - np.min(im))  # Normalization between 0-1
        x[i] = im
        names.append(name[2::])

    return x, names


def predict(test_data_dir, weights_path_list, out_seg_dir):
    padding= True   #should levels be padded
    depth= 5       #depth of the network
    wf= 5           #wf (int): number of filters in the first layer is 2**wf
    up_mode= 'upconv' #should we simply upsample the mask, or should we try and learn an interpolation
    batch_norm = True #should we use batch normalization between the layers
    batch_size = 1

    '''Read validation images'''
    x_val,names_val = load_train_imgs(test_data_dir)
    x_val = x_val[:,:,:,np.newaxis] # Add channel dimension

    '''Create validation dataset'''
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    dataset = Dataset(x_val, img_transform=img_transform)
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    '''Create and Load Model'''
    #specify if we should use a GPU (cuda) or only the CPU
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("You are using GPU")
    device = torch.device('cuda' if use_cuda else 'cpu')

    model_liver = UNet(n_classes=1, in_channels=1, padding=padding,depth=depth,wf=wf, up_mode=up_mode, batch_norm=batch_norm).to(device)
    model_lesion = UNet(n_classes=1, in_channels=2, padding=padding,depth=depth,wf=wf, up_mode=up_mode, batch_norm=batch_norm).to(device)
    # print(summary(model_liver,(in_channels,IMG_HEIGHT, IMG_WIDTH)))

    #----- generate output
    #load best model
    checkpoint = torch.load(weights_path_list[1])
    model_liver.load_state_dict(checkpoint["model_dict"])

    checkpoint = torch.load(weights_path_list[0])
    model_lesion.load_state_dict(checkpoint["model_dict"])

    if not os.path.exists(out_seg_dir):
        os.mkdir(out_seg_dir)

    print("Creating Segmentation Images..")

    model_liver.eval()
    model_lesion.eval()
    with torch.no_grad():
        for t, (X) in enumerate(dataLoader):
            X = X.to(device)                    # [Nbatch, 1, H, W]

            prediction_liver = model_liver(X)
            isLiver = (prediction_liver > 0.5).type(torch.cuda.FloatTensor)
            input_model_lesion = torch.cat((X, isLiver), 1)
            prediction_lesion = model_lesion(input_model_lesion)

            predLiver = prediction_liver.squeeze().cpu().numpy()
            predLiver = (predLiver > 0.5).astype(int)
            predLesion = prediction_lesion.squeeze().cpu().numpy()
            predLesion = (predLesion > 0.5).astype(int)
            segImage = np.maximum(predLiver * 127, predLesion * 255)

            imsave(os.path.join(out_seg_dir, 'myseg' + names_val[t] + '.png'), segImage, cmap='gray',vmin=0,vmax=255)

