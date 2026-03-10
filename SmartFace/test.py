import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from .data_util import imshow_pair
import matplotlib.pyplot as plt

def cal_fa_fr(dataloader, plane_size, model, device, margin):

    false_accept = []
    false_reject = []
    thresholds = np.linspace(0, margin * 2, 50)
    for threshold in tqdm(thresholds):
        fa = 0
        fr = 0
        total_samples = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                img0, img1 , label = data
                img0, img1 , label = img0.to(device), img1.to(device), label.to(device)
                zero_noise = (torch.zeros((1, plane_size, plane_size), device = device), torch.zeros((1, plane_size, plane_size), device = device))
                _, output1, _, output2, _, _ = model(img0,zero_noise, img1, zero_noise)
                euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
                pred_label = (euclidean_distance.cpu().numpy() > threshold).astype(int)
                label = label.cpu().numpy()
                total_samples += label.shape[0]
                for i in range(label.shape[0]):
                    if pred_label[i] != label[i]:
                        if label[i] == 0:
                            fr += 1
                        else:
                            fa += 1

        false_accept.append(fa/total_samples)
        false_reject.append(fr/total_samples)

    false_diff = np.abs(np.array(false_reject) - np.array(false_accept))
    false_sum = np.abs(np.array(false_reject) + np.array(false_accept))
    best_threshold = thresholds[np.argmin(false_diff)]
    lowest_rate = false_sum[np.argmin(false_diff)]/2
    print('best threshold:', best_threshold)
    print('lowest rate:', lowest_rate)
    out_data = {}
    out_data['thresholds'] = thresholds
    out_data['best_threshold'] = best_threshold
    out_data['lowest_rate'] = lowest_rate
    out_data['false_reject'] = false_reject
    out_data['false_accept'] = false_accept
    return out_data

def vis_random_test_samples(dataloader, plane_size, model, device, batch_size, writer, detector_paras):

        with torch.no_grad():
            data = next(iter(dataloader))
            img0, img1 , label = data
            img0, img1 , label = img0.to(device), img1.to(device), label.to(device)
            zero_noise = (torch.zeros((1, plane_size, plane_size), device = device), torch.zeros((1, plane_size, plane_size), device = device))
            signal1, output1, signal2, output2, meanI1, meanI2 = model(img0,zero_noise, img1, zero_noise)

            for batch_idx in tqdm(range(batch_size)):
                img1_sampled = img0[batch_idx, 0].cpu().numpy()
                img2_sampled = img1[batch_idx, 0].cpu().numpy()
                euclidean_distance = F.pairwise_distance(
                    output1[batch_idx], output2[batch_idx])
                face_fig = imshow_pair(img1_sampled, img2_sampled, title='Dissimilarity: {:.2f}, Label: {:.1f}'.format(
                    euclidean_distance.item(), label[batch_idx].item()), cmap='gray')
                writer.add_figure('TEST: Input face pair',
                                    face_fig,
                                    global_step=batch_idx)
                plt.close()
                output_img1 = signal1[batch_idx, 0].cpu().numpy()
                output_img2 = signal2[batch_idx, 0].cpu().numpy()

                output_fig = imshow_pair(output_img1, output_img2, detector_paras=detector_paras)
                writer.add_figure('TEST: Output intensity pair',
                                    output_fig,
                                    global_step=batch_idx)
                plt.close()
                
def save_all_pairs(dataloader, plane_size, model, device, out_path):
    dataloader = list(dataloader)
    out_path = os.path.join(out_path, "test_imgs_combination/")
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    total_samples = len(dataloader)
    for i in tqdm(range(total_samples - 1)):
        xi = dataloader[i]
        np.savetxt(os.path.join(out_path, f"face{i}.csv"), np.squeeze(xi.numpy()), delimiter=",")
        for j in range(i + 1, total_samples):
            xj = dataloader[j]
            np.savetxt(os.path.join(out_path, f"face{i}_{j}.csv"), np.squeeze(xj.numpy()), delimiter=",")
            with torch.no_grad():
                zero_noise = (torch.zeros((1, plane_size, plane_size), device = device), torch.zeros((1, plane_size, plane_size), device = device))
                signal1, output1,signal2, output2, _, _ = model(xi.to(device),zero_noise, xj.to(device), zero_noise)
                euclidean_distance = F.pairwise_distance(output1, output2)
                face_fig = imshow_pair(np.squeeze(xi.numpy()), np.squeeze(xj.numpy()), title='Dissimilarity: {:.2f}'.format(
                    euclidean_distance.item()), cmap='gray')
                face_fig.savefig(os.path.join(out_path, f"face{i}_{j}_pair.png"))
                plt.close()
                output_img1 = signal1.cpu().numpy()[0,0]
                output_img2 = signal2.cpu().numpy()[0,0]
                if j == i + 1:
                    np.savetxt(os.path.join(out_path, f"face{i}_field.csv"), output_img1, delimiter=",")
                np.savetxt(os.path.join(out_path, f"face{i}_{j}_feild.csv"), output_img2, delimiter=",")   
            
    # for i, data in enumerate(tqdm(dataloader)):
    #     img0, img1 , label = data
    #     np.savetxt(os.path.join(out_path, f"face{i}_0.csv"), np.squeeze(img0.numpy()), delimiter=",")
    #     np.savetxt(os.path.join(out_path, f"face{i}_1.csv"), np.squeeze(img1.numpy()), delimiter=",")
    #     img0_dev, img1_dev , label = img0.to(device), img1.to(device), label.to(device)
    #     with torch.no_grad():
    #         zero_noise = (torch.zeros((1, plane_size, plane_size), device = device), torch.zeros((1, plane_size, plane_size), device = device))
    #         signal1, output1,signal2, output2 = model(img0_dev,zero_noise, img1_dev, zero_noise)
    #         euclidean_distance = F.pairwise_distance(
    #             output1[0], output2[0])
    #         face_fig = imshow_pair(np.squeeze(img0.numpy()), np.squeeze(img1.numpy()), title='Dissimilarity: {:.2f}, Label: {:.1f}'.format(
    #             euclidean_distance.item(), label[0].item()), cmap='gray')
    #         face_fig.savefig(os.path.join(out_path, f"face{i}_pair.png"))
    #         output_img1 = signal1.cpu().numpy()[0,0]
    #         output_img2 = signal2.cpu().numpy()[0,0]
    #         if j == i + 1:
    #             np.savetxt(out_path + f"face{i}_field.csv", output_img1, delimiter=",")
    #         np.savetxt(out_path + f"face{i}_{j}_feild.csv", output_img2, delimiter=",")   