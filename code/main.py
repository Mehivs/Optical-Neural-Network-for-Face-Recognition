import sys
#module_path = '/home/zhicheng/Optical_Face_Recognition/'
#module_path = 'C:/Users/94735/OneDrive - UW-Madison/My Projects/Face_recognition2.0/stage2_SmartFaceGlass/'
import os
module_path = os.path.abspath("./")
sys.path.insert(1, module_path)
#the module path is the where SmartFace module is located.
import pandas as pd
from SmartFace import cal_fa_fr, save_all_pairs
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
from typing import List, Dict, Tuple, Optional
import argparse
import json
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torch.optim as optim
from SmartFace import Bipass_model, ContrastiveLoss, ContrastNormLoss
import torchvision.transforms as transforms
from SmartFace import Config, SiameseNetworkDataset, imshow_pair, simpleImgDataset, vis_random_test_samples
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

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

# look at Config dataclass in data_util for definition of each argument.
exp_config = Config(
    training_dir=json_config['training_dir'],
    testing_dir=json_config['testing_dir'],
    devices=json_config['devices'],
    num_layers=json_config['num_layers'],  # number of optical layer
    # the propagate distance for each layer, plus the objec to layer distance.
    prop_distances=tuple(json_config['prop_distances']),
    # the refractive index of the free space propagation.
    prop_mediums=tuple(json_config['prop_mediums']),
    grid=json_config['grid'],
    plane_size=json_config['plane_size'],
    train_batch_size=json_config['train_batch_size'],
    train_epoches=json_config['train_epoches'],
    prop_noise_dist=tuple(json_config['prop_noise_dist']),
    white_noise_dist=tuple(json_config['white_noise_dist']),
    mean_intensity_w=json_config['mean_intensity_w'],
    invert=json_config['invert'],  # invert image.
    # (counts for 1 line, size of detector)
    detector_paras=json_config['detector_paras'],
    margin=json_config['margin'],
    out_dir=json_config['out_dir'],
    log_freq=json_config['log_freq'],
    save_freq=json_config['save_freq'],
    pt_path=json_config['pt_path'],
    save_test_combinations=json_config['save_test_combinations'],
    num_workers=json_config['num_workers'],
    dummys=json_config['dummys'] if json_config.get('dummys') is not None else None,
    G_norm=json_config['G_norm'] if json_config.get('G_norm') is not None else 1,
    G_shifts=json_config['G_shifts'] if json_config.get('G_shifts') is not None else None,
    fcn_paras=json_config['fcn_paras'] if json_config.get('fcn_paras') is not None else None
)

    
def construct_model(num_layers: int, plane_size: int, grid: int, prop_distances: Tuple[float], G_norm: float, prop_mediums: Tuple[float], detector_paras: Tuple[float], dummys: Tuple[bool], G_shifts: Tuple[Tuple[float]], fcn_paras: Tuple[int]):

    model = Bipass_model(num_layers, dummys, plane_size, grid,
                          prop_distances, G_shifts,  G_norm, prop_mediums, detector_paras, fcn_paras)
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


dummys = [False] * \
    exp_config.num_layers if exp_config.dummys is None else exp_config.dummys
G_shifts = [(0, 0)] * len(exp_config.prop_distances) if exp_config.G_shifts is None else exp_config.G_shifts

prime_dev, model = construct_model(num_layers=exp_config.num_layers, plane_size=exp_config.plane_size, grid=exp_config.grid,
                                   prop_distances=exp_config.prop_distances, G_shifts=G_shifts, G_norm=exp_config.G_norm, prop_mediums=exp_config.prop_mediums, detector_paras=exp_config.detector_paras, dummys=dummys, fcn_paras=exp_config.fcn_paras)

# output dir
if not os.path.exists(exp_config.out_dir):
    os.mkdir(exp_config.out_dir)
out_path = os.path.join(exp_config.out_dir, notes)

if not os.path.exists(out_path):
    os.mkdir(out_path)

if exp_config.pt_path:
    checkpoint = torch.load(
        exp_config.pt_path, map_location=torch.device('cpu'))
    saved_state_dict = checkpoint['model_state_dict']
    #model.load_state_dict(checkpoint['model_state_dict'])
    state_dict = model.state_dict()
    for i in range(exp_config.num_layers):
        if not dummys[i]:
            state_dict['optical_model.opticals.' + str(i) + '.phase'] = saved_state_dict['optical_model.opticals.' + str(i) + '.phase'].to(
                state_dict['optical_model.opticals.' + str(i) + '.phase'].device)
    if exp_config.fcn_paras is not None:
        for key in saved_state_dict.keys():
            if key.startswith('fcn'):
                state_dict[key] = saved_state_dict[key]
    model.load_state_dict(state_dict)
    print('Load successed!')
else:
    model.apply(init_weights)

print(model)
writer = SummaryWriter(os.path.join(out_path, 'summary'))

# if train
if exp_config.training_dir:
    folder_dataset = dset.ImageFolder(root=exp_config.training_dir)

    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transforms.Compose([transforms.Resize((exp_config.plane_size, exp_config.plane_size)),
                                                                          transforms.ToTensor()
                                                                          ]), should_invert=exp_config.invert)

    #criterion = ContrastiveLoss(margin=exp_config.margin)
    #criterion = M.cosLoss()
    criterion = ContrastNormLoss(
        margin=exp_config.margin, mean_intensity_w=exp_config.mean_intensity_w)
    optimizer = optim.Adam(model.parameters())

    train_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  batch_size=exp_config.train_batch_size,
                                  num_workers=exp_config.num_workers)

    # noise sampler
    prop_noise_dist = torch.distributions.normal.Normal(
        *exp_config.prop_noise_dist)
    white_noise_dist = torch.distributions.normal.Normal(
        *exp_config.white_noise_dist)

    for epoch in tqdm(range(exp_config.train_epoches)):
        for i, data in enumerate(train_dataloader):
            img0, img1, label = data
            img0, img1, label = img0.to(prime_dev), img1.to(
                prime_dev), label.to(prime_dev)
            optimizer.zero_grad()
            noise1 = generate_noise(exp_config.plane_size,
                                    prop_noise_dist,  white_noise_dist, prime_dev)
            noise2 = generate_noise(exp_config.plane_size,
                                    prop_noise_dist,  white_noise_dist, prime_dev)
            #zero_noise = (torch.zeros((1, exp_config.plane_size, exp_config.plane_size), device = prime_dev), torch.zeros((1, exp_config.plane_size, exp_config.plane_size), device = prime_dev))
            #noise1 = noise2 = zero_noise
            signal1, output1, signal2, output2, meanI1, meanI2 = model(
                img0, noise1, img1, noise2)
            prime_loss, secondary_loss = criterion(
                output1, output2, label, meanI1, meanI2)
            total_loss = prime_loss + secondary_loss
            total_loss.backward()
            optimizer.step()

            global_step = epoch * len(train_dataloader) + i

            log_freq = exp_config.log_freq
            if (global_step) % log_freq == 0:
                writer.add_scalar('ContrastiveLoss',
                                  scalar_value=prime_loss.item(), global_step=global_step)
                writer.add_scalar('regularization',
                                  scalar_value=secondary_loss.item(), global_step=global_step)
                writer.add_scalar('totalLoss',
                                  scalar_value=total_loss.item(), global_step=global_step)
            save_freq = exp_config.save_freq
            if global_step % save_freq == 0:
                with torch.no_grad():
                    batch_idx = 0
                    img1_sampled = img0[batch_idx, 0].cpu().numpy()
                    img2_sampled = img1[batch_idx, 0].cpu().numpy()
                    euclidean_distance = F.pairwise_distance(
                        output1[batch_idx], output2[batch_idx])
                    face_fig = imshow_pair(img1_sampled, img2_sampled, title='Dissimilarity: {:.2f}, Label: {:.1f}'.format(
                        euclidean_distance.item(), label[batch_idx].item()), cmap='gray')
                    writer.add_figure('Input face pair',
                                      face_fig,
                                      global_step=global_step)
                    plt.close()
                    output_img1 = signal1[batch_idx, 0].cpu().numpy()
                    output_img2 = signal2[batch_idx, 0].cpu().numpy()

                    output_fig = imshow_pair(output_img1, output_img2, detector_paras=exp_config.detector_paras)
                    writer.add_figure('Output intensity pair',
                                      output_fig,
                                      global_step=global_step)
                    plt.close()
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(out_path, 'checkpoint_step_' + str(global_step) + '.pt'))

    writer.add_graph(model, (img0, noise1, img1, noise2))
    for i in range(exp_config.num_layers):
        if not dummys[i]:
            phasei = np.squeeze(model.optical_model.opticals[i].phase.cpu().detach().numpy())
            plt.figure()
            plt.imshow(phasei)
            plt.colorbar()
            plt.savefig(os.path.join(out_path, f"Phase_of_layer_{i}.png"))
            plt.close()
            np.savetxt(os.path.join(
                out_path, f"Phase_of_layer_{i}.csv"), phasei, delimiter=',')

# if test
if exp_config.testing_dir:
    folder_dataset_test = dset.ImageFolder(root=exp_config.testing_dir)
    siamese_dataset_test = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                                 transform=transforms.Compose([transforms.Resize((exp_config.plane_size, exp_config.plane_size)),
                                                                               transforms.ToTensor()
                                                                               ]), should_invert=False)
    test_dataloader = DataLoader(siamese_dataset_test, batch_size=exp_config.train_batch_size,
                                 shuffle=False, num_workers=exp_config.num_workers)

    vis_random_test_samples(test_dataloader, exp_config.plane_size,
                            model, prime_dev, exp_config.train_batch_size, writer, exp_config.detector_paras)

    out_data = cal_fa_fr(
        test_dataloader, exp_config.plane_size, model, prime_dev, exp_config.margin)
    out_df = pd.DataFrame.from_dict(out_data)
    out_df.to_csv(os.path.join(out_path, "test_set_FR_FA.csv"))
    thresholds = out_data['thresholds']
    FR_FA_fig = plt.figure()
    plt.plot(thresholds, out_data['false_reject'], label='False reject')
    plt.plot(thresholds, out_data['false_accept'], label='False accept')
    #plt.plot(thresholds, out_data['false_reject'].values + out_data['false_accept'].values, label='Total error rate', c = '#C00000', linestyle = '--')
    plt.ylim(-0.02, 0.6)
    plt.legend()
    plt.title(
        f"Best threshold: {out_data['best_threshold']:.2f}, rate: {out_data['lowest_rate']:.2f}")
    plt.savefig(os.path.join(out_path, f"FR_FA_with_sum_acc.png"))
    writer.add_figure('False reject and false accept plot',
                      FR_FA_fig,
                      global_step=0)
    plt.close()


if exp_config.save_test_combinations:
    folder_dataset_test = dset.ImageFolder(root=exp_config.testing_dir)
    siamese_dataset_test = simpleImgDataset(imageFolderDataset=folder_dataset_test,
                                            transform=transforms.Compose([transforms.Resize((exp_config.plane_size, exp_config.plane_size)),
                                                                          transforms.ToTensor()
                                                                          ]), should_invert=False)
    test_dataloader = DataLoader(
        siamese_dataset_test, batch_size=1, shuffle=False)
    save_all_pairs(test_dataloader, exp_config.plane_size,
                   model, prime_dev, out_path)

writer.close()
