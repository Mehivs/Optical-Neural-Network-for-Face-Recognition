import sys
#module_path = '/home/zhicheng/Optical_Face_Recognition/'
module_path = 'C:/Users/94735/OneDrive - UW-Madison/My Projects/Face_recognition2.0/stage2_SmartFaceGlass/'
sys.path.insert(1, module_path)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from SmartFace import Config, simpleImgDataset, imshow_pair, ContrastNormLoss, lens_profile
import torchvision.transforms as transforms
from SmartFace import optical_model
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import json
import argparse
from typing import Tuple
import torch.nn.functional as F
# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--config', type=str, required=True)
# Parse the argument
args = parser.parse_args()
notes = args.config.split('/')[-1][:-5]
print(f"the notes is: f{notes}, which will be used as ouput dir name.")

with open(args.config, "r") as jsonfile:
    json_config = json.load(jsonfile)
    print("Json config read successful.")

enlarge_ratio = 2
# look at Config dataclass in data_util for definition of each argument.
exp_config = Config(
    training_dir=json_config['training_dir'],
    testing_dir=json_config['testing_dir'],
    devices=json_config['devices'],
    num_layers=json_config['num_layers'],  # number of optical layer
    # the propagate distance for each layer, plus the objec to layer distance.
    prop_distances=tuple([dis * enlarge_ratio for dis in json_config['prop_distances']]),
    # the refractive index of the free space propagation.
    prop_mediums=tuple(json_config['prop_mediums']),
    grid=json_config['grid'],
    plane_size=json_config['plane_size'] * enlarge_ratio,
    train_batch_size=json_config['train_batch_size'],
    train_epoches=json_config['train_epoches'],
    prop_noise_dist=tuple(json_config['prop_noise_dist']),
    white_noise_dist=tuple(json_config['white_noise_dist']),
    mean_intensity_w=json_config['mean_intensity_w'],
    invert=json_config['invert'],  # invert image.
    # (counts for 1 line, size of detector)
    detector_paras=(json_config['detector_paras'][0],json_config['detector_paras'][1] * enlarge_ratio),
    margin=json_config['margin'],
    out_dir=json_config['out_dir'],
    log_freq=json_config['log_freq'],
    save_freq=json_config['save_freq'],
    pt_path=json_config['pt_path'],
    save_test_combinations=json_config['save_test_combinations'],
    num_workers=json_config['num_workers']
)


def construct_model(num_layers: int, plane_size: int, grid: int, prop_distances: Tuple[float], prop_mediums: Tuple[float], detector_paras: Tuple[float]):

    model = optical_model(num_layers, plane_size, grid,
                            prop_distances, prop_mediums, detector_paras)
    if type(exp_config.devices) == str:
        prime_dev = torch.device(exp_config.devices)
        model.to(prime_dev)
    else:
        # Handle multi-gpu if desired
        prime_dev = torch.device(exp_config.devices[0])
        model.to(prime_dev)
        model = nn.DataParallel(model, exp_config.devices)

    return prime_dev, model


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('optical_layer') != -1:
        m.reset()


def generate_noise(plane_size, prop_noise_dist,  white_noise_dist, device):

    prop_noise = prop_noise_dist.sample(torch.Size(
        [1, plane_size, plane_size])).to(device, dtype=torch.float)
    white_noise = white_noise_dist.sample(torch.Size(
        [1, plane_size, plane_size])).to(device, dtype=torch.float)
    noise = (prop_noise, white_noise)
    return noise


prime_dev, model = construct_model(num_layers=exp_config.num_layers, plane_size=exp_config.plane_size, grid=exp_config.grid,
                                   prop_distances=exp_config.prop_distances, prop_mediums=exp_config.prop_mediums, detector_paras=exp_config.detector_paras)


def enlarge(T: torch.Tensor, enlarge_ratio: float) -> torch.Tensor:
    #dev = T.device
    out_size = (int(T.shape[2] * enlarge_ratio), int(T.shape[2] * enlarge_ratio))
    return torch.nn.functional.interpolate(T, size = out_size)
    
if exp_config.pt_path:
    checkpoint = torch.load(exp_config.pt_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['model_state_dict']
    cur_state_dict = model.state_dict()
    for i in range(exp_config.num_layers):
        cur_state_dict['opticals.' + str(i) + '.phase'] = enlarge(state_dict['opticals.' + str(i) + '.phase'], enlarge_ratio)
    model.load_state_dict(cur_state_dict)
    print('Intialized by enlarged metasurface.')
else:
    model.apply(init_weights)

initial_by_lens = False

if initial_by_lens:
    lens = lens_profile(exp_config.plane_size, exp_config.grid, exp_config.prop_distances[0]/2, 1)
    lens = torch.tensor(lens, dtype = torch.float32)
    state_dict = model.state_dict()
    for i in range(exp_config.num_layers):
        state_dict['opticals.' + str(i) + '.phase'] = lens.to(state_dict['opticals.' + str(i) + '.phase'].device)
    model.load_state_dict(state_dict)
    print('Intialized by lens.')
print(model)
criterion = ContrastNormLoss(margin=exp_config.margin, mean_intensity_w=exp_config.mean_intensity_w)
# noise sampler
prop_noise_dist = torch.distributions.normal.Normal(
    *exp_config.prop_noise_dist)
white_noise_dist = torch.distributions.normal.Normal(
    *exp_config.white_noise_dist)

idx = 0
folder_dataset_test = dset.ImageFolder(root=exp_config.testing_dir)
siamese_dataset_test = simpleImgDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Resize((exp_config.plane_size, exp_config.plane_size)),
                                                                        transforms.ToTensor()
                                                                        ]), should_invert=False)
test_dataloader = DataLoader(
    siamese_dataset_test, batch_size=1, shuffle=False)
test_dataloader = list(test_dataloader)
xi = test_dataloader[idx]
xj = test_dataloader[idx + 26]
label = torch.ones((1,1), dtype = torch.float)
with torch.no_grad():
    noise1 = generate_noise(exp_config.plane_size,
                            prop_noise_dist,  white_noise_dist, prime_dev)
    noise2 = generate_noise(exp_config.plane_size,
                            prop_noise_dist,  white_noise_dist, prime_dev)
    #zero_noise = (torch.zeros((1, exp_config.plane_size, exp_config.plane_size), device = prime_dev), torch.zeros((1, exp_config.plane_size, exp_config.plane_size), device = prime_dev))
    signal1, output1,signal2, output2, meanI1, meanI2 = model(xi.to(prime_dev),noise1, xj.to(prime_dev), noise2)

    prime_loss, secondary_loss = criterion(output1, output2, label.to(prime_dev), meanI1, meanI2)
    total_loss = prime_loss + secondary_loss
    print(f"prime loss: {prime_loss.item():2f}")
    print(f"secondary loss: {secondary_loss.item():2f}")
    print(f"total loss: {total_loss.item():2f}")
    euclidean_distance = F.pairwise_distance(output1, output2)
    face_fig = imshow_pair(np.squeeze(xi.numpy()), np.squeeze(xj.numpy()), title='Dissimilarity: {:.2f}'.format(
        euclidean_distance.item()), cmap='gray')
    plt.show()
    output_img1 = signal1.cpu().numpy()[0,0]
    output_img2 = signal2.cpu().numpy()[0,0]

output_fig = imshow_pair(output_img1, output_img2)
plt.show()

