from unet import UNet

import os # os specific actions
import numpy as np # linear algebra
import cv2 # image processing
import matplotlib.pyplot as plt # plots
import PIL
import random
import sys
import math

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary

import time


class Dataset(object):
    def __init__(self, x, y1, y2, img_transform=None, mask_transform=None):
        self.len = len(y1)
        self.x_data = x
        self.y1_data = y1
        self.y2_data = y2
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.y1_data)

    def __getitem__(self, index):
        img = self.x_data[index]
        mask1 = self.y1_data[index]
        mask2 = self.y2_data[index]

        seed = random.randrange(sys.maxsize)

        if self.img_transform:
            random.seed(seed)  # apply this seed to img transforms
            img_new = self.img_transform(img)

        if self.mask_transform:
            random.seed(seed)
            mask_new1 = self.mask_transform(mask1)
            mask_new1 = np.asarray(mask_new1).squeeze()
            random.seed(seed)
            mask_new2 = self.mask_transform(mask2)
            mask_new2 = np.asarray(mask_new2).squeeze()

        return img_new, mask_new1, mask_new2

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
def load_train_imgs(path_IMAGE_LIB,path_MASK_LIB):
    train_images_ct = [x for x in sorted(os.listdir(path_IMAGE_LIB)) if x[-4:] == '.png']  # Read all the images
    train_images_seg = [x for x in sorted(os.listdir(path_MASK_LIB)) if x[-4:] == '.png']  # Read all the images

    IMG_HEIGHT, IMG_WIDTH = 512, 512

    x = np.empty((len(train_images_ct), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
    for i, name in enumerate(train_images_ct):
        im = cv2.imread(path_IMAGE_LIB + name, cv2.IMREAD_GRAYSCALE).astype('int16').astype('float32')
        im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
        im = (im - np.min(im)) / (np.max(im) - np.min(im))  # Normalization between 0-1
        x[i] = im

    y_liver = np.empty((len(train_images_seg), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
    for i, name in enumerate(train_images_seg):
        im = cv2.imread(path_MASK_LIB + name, cv2.IMREAD_GRAYSCALE).astype('float32') / 255.
        im[im > 0] = 1  # unite liver and lesion segmentation
        im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
        y_liver[i] = im

    y_lesion = np.empty((len(train_images_seg), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
    for i, name in enumerate(train_images_seg):
        im = cv2.imread(path_MASK_LIB + name, cv2.IMREAD_GRAYSCALE).astype('float32') / 255.
        im[im < 1] = 0  # lesion segmentation
        im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
        y_lesion[i] = im

    return x, y_liver, y_lesion



def train(train_data_dir, val_data_dir, out_weights_dir):

    train_path_IMAGE_LIB = train_data_dir + 'ct/'
    val_path_IMAGE_LIB = val_data_dir + 'ct/'
    train_path_MASK_LIB = train_data_dir + 'seg/'
    val_path_MASK_LIB = val_data_dir + 'seg/'

    IMG_HEIGHT, IMG_WIDTH = 512, 512

    phases = ["train","val"]

    n_classes= 2    #number of classes in the data mask that we'll aim to predict
    in_channels_liver = 1  #input channel of the data, RGB = 3, Gray = 1
    in_channels_lesion = 2
    padding= True   #should levels be padded
    depth= 5       #depth of the network
    wf= 5           #wf (int): number of filters in the first layer is 2**wf
    up_mode= 'upconv' #should we simply upsample the mask, or should we try and learn an interpolation
    batch_norm = True #should we use batch normalization between the layers

    patch_size = IMG_HEIGHT
    batch_size = 4
    num_epochs = 50

    x_train, y_train_liver, y_train_lesion = load_train_imgs(train_path_IMAGE_LIB,train_path_MASK_LIB)
    x_val, y_val_liver, y_val_lesion = load_train_imgs(val_path_IMAGE_LIB,val_path_MASK_LIB)


    x_train = x_train[:,:,:,np.newaxis] # Add channel dimension
    y_train_liver = y_train_liver[:,:,:,np.newaxis]
    y_train_lesion = y_train_lesion[:,:,:,np.newaxis]
    x_val = x_val[:,:,:,np.newaxis] # Add channel dimension
    y_val_liver = y_val_liver[:,:,:,np.newaxis]
    y_val_lesion = y_val_lesion[:,:,:,np.newaxis]

    '''
    make dataloaders for train and validation datasets + AUGMENTATION
    '''
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(size=(patch_size, patch_size), pad_if_needed=True),
        transforms.RandomResizedCrop(size=patch_size, scale=(0.9, 1)),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(size=(patch_size, patch_size), pad_if_needed=True),
        transforms.RandomResizedCrop(size=patch_size, scale=(0.9, 1), interpolation=PIL.Image.NEAREST),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    dataset = {}
    dataLoader = {}
    for phase in phases:
        if phase == 'train':
            print("Load Train Images")
            dataset[phase] = Dataset(x_train, y_train_liver, y_train_lesion, img_transform=img_transform, mask_transform=mask_transform)
            dataLoader[phase] = DataLoader(dataset[phase], batch_size=batch_size, shuffle=True)
        elif phase == 'val':
            print("Load Validation Images")
            dataset[phase] = Dataset(x_val, y_val_liver, y_val_lesion, img_transform=img_transform, mask_transform=mask_transform)
            dataLoader[phase] = DataLoader(dataset[phase], batch_size=batch_size, shuffle=True)


    '''define our model'''
    # specify if we should use a GPU (cuda) or only the CPU
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("You are using GPU")
    device = torch.device('cuda' if use_cuda else 'cpu')

    #build the model according to the parameters specified above and copy it to the GPU. finally print out the number of trainable parameters
    model_liver = UNet(n_classes=1, in_channels=1, padding=padding,depth=depth,wf=wf, up_mode=up_mode, batch_norm=batch_norm).to(device)
    model_lesion = UNet(n_classes=1, in_channels=2, padding=padding,depth=depth,wf=wf, up_mode=up_mode, batch_norm=batch_norm).to(device)
    # print(summary(model_liver,(2,IMG_HEIGHT, IMG_WIDTH)))

    param = list(model_liver.parameters()) + list(model_lesion.parameters())
    optim = torch.optim.Adam(param)
    criterion = nn.BCELoss()

    '''start training'''
    print("Start Training..")
    all_losses_train = np.zeros(num_epochs)
    all_losses_val = np.zeros(num_epochs)

    best_loss_on_test = np.Infinity
    start_time = time.time()
    for epoch in range(num_epochs):
        all_loss = {key: torch.zeros(0).to(device) for key in phases}

        for phase in phases: # iterate through both training and validation states

            if phase == 'train':
                model_liver.train()  # Set model to training mode
                model_lesion.train()
            else:                    # when in eval mode, we don't want parameters to be updated
                model_liver.eval()   # Set model to evaluate mode
                model_lesion.eval()

            for ii , (X, y_liver, y_lesion) in enumerate(dataLoader[phase]): # for each of the batches
                X = X.to(device)                                        # [Nbatch, 1, H, W]
                y_liver = y_liver.type('torch.FloatTensor').to(device)  # [Nbatch, H, W] with class indices (0, 1)
                y_lesion = y_lesion.type('torch.FloatTensor').to(device)

                with torch.set_grad_enabled(phase == 'train'): # dynamically set gradient computation, in case of validation, this isn't needed
                                                                # disabling is good practice and improves inference time

                    prediction_liver = model_liver(X)  # [N, Nclass, H, W]
                    input_model_lesion = torch.cat((X, prediction_liver), 1)
                    prediction_lesion = model_lesion(input_model_lesion)

                    prediction_liver_flat = prediction_liver.view(-1)
                    prediction_lesion_flat = prediction_lesion.view(-1)
                    y_liver_flat = y_liver.view(-1)
                    y_lesion_flat = y_lesion.view(-1)

                    loss_liver = criterion(prediction_liver_flat, y_liver_flat)
                    loss_lesion = criterion(prediction_lesion_flat, y_lesion_flat)
                    loss = loss_liver + loss_lesion

                    if phase=="train": #in case we're in train mode, need to do back propagation
                        optim.zero_grad()
                        loss.backward()
                        optim.step()

                    all_loss[phase]=torch.cat((all_loss[phase],loss.detach().view(1,-1)))

            all_loss[phase] = all_loss[phase].cpu().numpy().mean()

        print('%s ([%d/%d] %d%%), train loss: %.4f val loss: %.4f' % (timeSince(start_time, (epoch+1) / num_epochs), epoch+1, num_epochs ,(epoch+1) / num_epochs * 100, all_loss["train"], all_loss["val"]),end="")

        all_losses_train[epoch] = all_loss["train"]
        all_losses_val[epoch] = all_loss["val"]

        # if current loss is the best we've seen, save model state with all variables
        # necessary for recreation
        if all_loss["val"] < best_loss_on_test:
            best_loss_on_test = all_loss["val"]
            print("  **")
            state_liver = {'epoch': epoch + 1,
             'model_dict': model_liver.state_dict(),
             'optim_dict': optim.state_dict(),
             'best_loss_on_test': all_loss,
             'n_classes': n_classes,
             'in_channels': in_channels_liver,
             'padding': padding,
             'depth': depth,
             'wf': wf,
             'up_mode': up_mode, 'batch_norm': batch_norm}
            state_lesion = {'epoch': epoch + 1,
             'model_dict': model_lesion.state_dict(),
             'optim_dict': optim.state_dict(),
             'best_loss_on_test': all_loss,
             'n_classes': n_classes,
             'in_channels': in_channels_lesion,
             'padding': padding,
             'depth': depth,
             'wf': wf,
             'up_mode': up_mode, 'batch_norm': batch_norm}


            torch.save(state_liver, out_weights_dir + "weights_liver.pth")
            torch.save(state_lesion, out_weights_dir + "weights_lesion.pth")
        else:
            print("")



    '''Plot graph of train loss and val loss'''
    '''
    plt.plot(all_losses_train, color='b',marker='o',markersize=10,label='Train')
    plt.plot(all_losses_val, color='r',marker='o',markersize=10,label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Cross entropy loss")
    plt.savefig('loss_graph_SGD.png')
    plt.show()
    '''

