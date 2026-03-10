import numpy as np
import cv2
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
from PIL import Image
import PIL.ImageOps
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union


WAVELENGTH = 1

@dataclass
class Config:
    training_dir: Optional[str] #if is None then not train model.
    testing_dir: Optional[str] #if is None then not to test model.
    devices: Union[List[str], str] # "cuda" or "cpu" or a list of cudas. e.g. ['cuda_0', 'cuda_1']
    num_layers: int #number of optical layer.note: when go from free space into substrate, you are passing 1 layer. To simulate a matasurface with substrate you need num_layer = 2.
    
    # the propagate distance for each layer.
    prop_distances: Tuple[float]
    # the refractive index of the free space propagation.
    prop_mediums: Tuple[float]
    grid: int
    plane_size: int
    train_batch_size: int
    train_epoches: int
    prop_noise_dist: Tuple[float] #(mu, sigma)
    white_noise_dist: Tuple[float] # (mu, sigma) of white noise
    mean_intensity_w: float #the regulazation weight in the contrastNormLoss.
    invert: bool #invert the color of the image. e.g. gray scale pixels in range [0, 1] will be invert to [1, 0]. value 0.2 will be come 0.8
    
    #note: if detector_size <= 0: then the detector_size will be calculated as plane_size//counts
    detector_paras: Tuple[int] #(counts, detector_size) e.g. (3, 15), there are 3 * 3 detectors, each with size 15 * 15 pixels.
    margin: float #the parameter used in loss definition.
    out_dir: str #output floder save everything. 
    log_freq: int #log loss per number of steps.
    save_freq: int #save image per number of steps 
    pt_path: Optional[str] #the path to load pre_trained model. e.g.:  "debug/output/test/checkpoint_step_1359.pt", if the path is null, default initialization is applied.
    save_test_combinations: bool #if true, all combination of test image pair will be saved in folder: test_imgs_combination, will be super slow if you have a lot of images.
    num_workers: int #num_workers for data loader.
    dummys: Tuple[bool] #if the optical layer is dummy, it means no phase and amplitude modification is added.
    G_norm: float #the free space propagator is normalized by G_norm to conserve energy.
    G_shifts: Tuple[Tuple[float]] #G_shifts[0] means shift the input. G_shifts[0] = (shift_x, shift_y) [lam]
    fcn_paras: Tuple[int] #the paras for fully connected network (in_size, out_size, layers, nodes). if layers == 0, there is only 1 hidden layer. nodes always > 0. The total hidden lyaer = layers + 1.
    def __post_init__(self):
        assert self.detector_paras[0] * self.detector_paras[1] <= self.plane_size, "the detector size is too large to fit in the plane size."
        assert self.num_layers + 1 == len(self.prop_distances), "the number of prop distances should equal to the (number layers + 1)."
        assert self.num_layers + 1 == len(self.prop_mediums), "the number of prop mediums should equal to the (number layers + 1)."
        if self.dummys:
            assert self.num_layers == len(self.dummys), "the number of dummy indicator should equal to the number of layers."
        if self.G_shifts:
            assert len(self.prop_distances) == len(self.G_shifts), "the number of propagator shifts equal to the number of prop distances."
        if self.fcn_paras:
            assert self.fcn_paras[0] == self.detector_paras[0]**2, "the in_size of fcn should equal to number of detectors."
#sample all possible pairs.
class AllPairImgDataset(Dataset):
    #SiameseNetworkDataset will make sure 50% chance the imags pair are from the same person.
    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        i = index//len(self.imageFolderDataset.imgs)
        j = index%len(self.imageFolderDataset.imgs)
        img0_tuple = self.imageFolderDataset.imgs[i]
        img1_tuple = self.imageFolderDataset.imgs[j]
        
        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        # transform image to grayscale
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        # invert the color, out = MAX - images
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)**2

class simpleImgDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        img0_tuple = self.imageFolderDataset.imgs[index]
        img0 = Image.open(img0_tuple[0])
        #transform image to grayscale
        img0 = img0.convert("L")
        
        #invert the color, out = MAX - images
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)

        if self.transform is not None:
            img0 = self.transform(img0)
        
        return img0
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)
    
class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        # transform image to grayscale
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        # invert the color, out = MAX - images
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

def imshow_pair(img1, img2, title = None, cmap = 'viridis', detector_paras = None):
    plane_size = img1.shape[0]
    if detector_paras is not None and detector_paras[1] > 0:
        counts = detector_paras[0]
        detector_size = detector_paras[1]
        label2pos = {}
        for i in range(counts):
            for j in range(counts):
                label2pos[i*counts + j] = (plane_size//(
                    counts + 1) * (i + 1), plane_size//(counts + 1) * (j + 1))
        sigma = detector_size//2
        color1 = img1.max() * 0.9
        color2 = img2.max() * 0.9
        for i in range(counts**2):
            pt1 = (label2pos[i][0] - sigma, label2pos[i][1] - sigma)
            pt2 = (label2pos[i][0] + sigma, label2pos[i][1] + sigma)
            img1 = cv2.rectangle(img1, pt1, pt2, color = color1, thickness= 1)
            img2 = cv2.rectangle(img2, pt1, pt2, color = color2, thickness= 1)
    if detector_paras is not None and detector_paras[1] <= 0:
        counts = detector_paras[0]
        detector_size = plane_size//counts
        color1 = img1.max() * 0.9
        color2 = img2.max() * 0.9
        for i in range(1, counts):
            img1[i * detector_size, :] = color1
            img1[:, i * detector_size] = color1
            img2[i * detector_size, :] = color2
            img2[:, i * detector_size] = color2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))
    if title:
        fig.suptitle(title, fontsize = 20)
    plot1 = ax1.imshow(img1, cmap = cmap)
    plt.colorbar(plot1, ax = ax1)
    plot2 = ax2.imshow(img2, cmap = cmap)
    plt.colorbar(plot2, ax = ax2)
    #plt.show()
    return fig
def imshow(img,text=None, out_path = None):
    npimg = img.cpu().numpy()
    fig = plt.figure()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0))) 
    if out_path:
        plt.savefig(out_path) 
    return fig
        
def save_npy(img, path):
    np.save(path, img)
    return


def save_batch(name, img_batch, pred_label, label, gl_step, image_path, npy_path=False):

    for i in range(img_batch.shape[0]):
        toi_path = image_path + gl_step + 'index_' + str(i) + name + '.png'
        toi_title = 'predict_label: ' + \
            str(pred_label[i]) + '_true_label: ' + str(label[i])
        save_image(img_batch[i], toi_path, toi_title, True)

        if npy_path:
            path = npy_path + gl_step + 'index_' + str(i) + name + '.npy'
            save_npy(img_batch[i], path)

    return

def save_image(img, path, title, colorbar):
    fig = plt.figure()
    plt.imshow(img)
    if colorbar:
        plt.colorbar()
        plt.grid(True)
    plt.title(title)
    plt.savefig(path)
    plt.close(fig)
    return

def save_to_excel(data, name, plane_size, path):

    data = np.reshape(data, (plane_size, plane_size))
    data_df = pd.DataFrame(data)
    data_df.to_excel(path + 'excel_data/' + name +
                     '.xlsx', index=False, header=False)


# def CM(y_test, y_test_pred, classes, path):

#     cm = confusion_matrix(y_test, y_test_pred, labels=np.arange(classes))
#     # Calculate and show correlation matrix
#     sns.set(font_scale=1)
#     hm = sns.heatmap(cm,
#                      cbar=True,
#                      annot=True,
#                      square=True,
#                      fmt='d',
#                      annot_kws={'size': 8},
#                      yticklabels=np.arange(classes),
#                      xticklabels=np.arange(classes))
#     plt.xlabel('prediction')
#     plt.ylabel('ground_truth')
#     plt.savefig(path+'/' + 'confusion_matrix.png')


