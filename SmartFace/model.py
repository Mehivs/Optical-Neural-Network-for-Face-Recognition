import torch
import torch.nn.functional as F
import torch.fft
from .optical_util import propagator
from .data_util import WAVELENGTH
import numpy as np


def fourier_conv(signal: torch.Tensor, f_kernel: torch.Tensor) -> torch.Tensor:
    '''
        args:
        signal, kernel: complex tensor, assume the images are square. the last 2 dim of signal is the height, and width of images.
    '''
    s_size = signal.size()
    k_size = f_kernel.size()
    padding = (k_size[-1] - s_size[-1])//2
    if (k_size[-1] - s_size[-1]) % 2 == 0:
        signal = F.pad(signal, (padding, padding, padding, padding))
    else:
        signal = F.pad(signal, (padding, padding + 1, padding, padding + 1))

    f_signal = torch.fft.fftn(signal, dim=(-2, -1))

    f_output = f_signal * f_kernel
    f_output = torch.fft.ifftn(f_output, dim=(-2, -1))
    f_output = f_output[:, :, padding: padding +
                        s_size[-1], padding:padding + s_size[-1]]

    return f_output

class dummy_layer(torch.nn.Module):
    def __init__(self,):
        super(dummy_layer, self).__init__()
    def forward(self, signal):
        return signal
    def reset(self):
        return 
    
class optical_layer(torch.nn.Module):
    def __init__(self, plane_size):
        '''

        '''
        super(optical_layer, self).__init__()
        self.plane_size = plane_size
        self.phase = torch.nn.Parameter(torch.empty(
            (1, 1, plane_size, plane_size), dtype=torch.float))

    def forward(self, signal):
        '''
            f: torch.nn.functional
        '''
        phase_real = torch.cos(self.phase)
        phase_imag = torch.sin(self.phase)
        c_phase = torch.complex(phase_real, phase_imag)
        signal = signal * c_phase

        return signal

    def reset(self):
        #nn.init_normal_(self.phase, 0, 0.02)
        torch.nn.init.constant_(self.phase, val=0)

class FCN_model(torch.nn.Module):
    def __init__(self, in_size: int, out_size: int, layers: int=2, nodes: int=64):
        super(FCN_model, self).__init__()
        module_list = [torch.nn.Linear(in_size, nodes), torch.nn.ReLU()]
        for _ in range(layers):
            module_list.append(torch.nn.Linear(nodes, nodes))
            module_list.append(torch.nn.ReLU())
        module_list.append(torch.nn.Linear(nodes, out_size))
        module_list.append(torch.nn.Sigmoid())
        self.fc = torch.nn.Sequential(*module_list)

    def forward(self, x):
        return self.fc(x)
    
class Bipass_model(torch.nn.Module):
    def __init__(self, num_layers, dummys, plane_size, grid, prop_distances, G_shifts, G_norm, prop_mediums, detector_paras, fcn_paras):
        super(Bipass_model, self).__init__()
        self.optical_model = optical_model(num_layers, dummys, plane_size, grid, prop_distances, G_shifts, G_norm, prop_mediums, detector_paras)
        self.fcn = None
        if fcn_paras is not None:
            in_size, out_size, layers, nodes = fcn_paras
            self.fcn = FCN_model(in_size, out_size, layers, nodes)
    
    def forward_once(self, signal, noise):
        signal, detector, mean_intensity = self.optical_model(signal, noise)
        if self.fcn is not None:
            detector = self.fcn(detector)
        return signal, detector, mean_intensity
    
    def forward(self, signal1, noise1, signal2, noise2):
        signal1, detector1, mean_intensity1 = self.forward_once(signal1, noise1)
        signal2, detector2, mean_intensity2 = self.forward_once(signal2, noise2)
        return signal1, detector1, signal2, detector2, mean_intensity1, mean_intensity2

class optical_model(torch.nn.Module):
    def __init__(self, num_layers, dummys, plane_size, grid, prop_distances, G_shifts, G_norm, prop_mediums, detector_paras):
        '''
        args:
            sensor: sensor size
        '''
        super(optical_model, self).__init__()
        self.plane_size = plane_size
        self.num_layers = num_layers
        opticals = []
        for i in range(self.num_layers):
            if dummys[i]:
                opticals.append(dummy_layer())
            else:
                opticals.append(optical_layer(plane_size))
        self.opticals = torch.nn.ModuleList(opticals)
        # len(prop_distances) = num_layers  + 1= len(prop_mediums)
        for i in range(len(prop_distances)):
            prop = propagator(plane_size, grid,
                              prop_distances[i], WAVELENGTH/prop_mediums[i], G_norm, G_shifts[i])
            f_kernel = np.fft.fft2(np.fft.ifftshift(prop))
            f_kernel = torch.tensor(f_kernel, device='cpu',
                                    dtype=torch.complex64, requires_grad=False)
            self.register_buffer(f'fk_const_{i}', f_kernel)

        self.counts = detector_paras[0]
        self.detector_size = detector_paras[1]
        self.label2pos = {}
        for i in range(self.counts):
            for j in range(self.counts):
                self.label2pos[i*self.counts + j] = (plane_size//(
                    self.counts + 1) * (i + 1), plane_size//(self.counts + 1) * (j + 1))
        #self.sm = torch.nn.Softmax(dim = -1)
        
    def forward(self, signal, noise):

        # object free space prop to reach optical structure
        signal = fourier_conv(signal, getattr(self, f'fk_const_0'))
        for i in range(self.num_layers):
            # phase modulation
            signal = self.opticals[i](signal)
            signal = fourier_conv(signal, getattr(self, f'fk_const_{i + 1}'))

        signal = signal.abs()**2
        prop_noise = noise[0]
        white_noise = noise[1]
        signal = signal * (1 + prop_noise) + white_noise
        
        if self.detector_size > 0:  
            #squared detectors with boundary.  
            sigma = self.detector_size//2
            detectors = [torch.mean(signal[:, :, self.label2pos[i][0] - sigma:self.label2pos[i][0] + sigma,
                                        self.label2pos[i][1] - sigma: self.label2pos[i][1] + sigma], dim=(2, 3)) for i in range(self.counts**2)]
        else:
            #whole detector area.
            detector_size = self.plane_size//self.counts
            detectors = []
            for i in range(self.counts):
                for j in range(self.counts):
                    detectors.append(torch.mean(signal[:,:,i * detector_size: (i + 1) * detector_size, j * detector_size : (j + 1) * detector_size], dim=(2, 3)))
        
        detectors = torch.cat(detectors, dim=-1)
        mean_intensity = detectors.mean()
        #detectors = torch.nn.functional.normalize(detectors, p=1.0, dim=-1, eps=1e-12, out=None)
        max_detector = torch.max(detectors, dim= -1, keepdim= True).values
        detectors = detectors / max_detector
        return signal, detectors, mean_intensity



class ContrastNormLoss(torch.nn.Module):
    """
    Contrastive loss function with output normalization.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0, mean_intensity_w = 0.1):
        super(ContrastNormLoss, self).__init__()
        self.margin = margin
        self.mean_intensity_w = mean_intensity_w
        #self.intensity_margin = None
    def forward(self, output1, output2, label, mean_intensity1, mean_intensity2):
        #detectors = self.sm(detectors)
        mean_intensity = (mean_intensity1 + mean_intensity2)/2
        euclidean_distance = F.pairwise_distance(
            output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        #loss = loss_contrastive + torch.clamp(self.intensity_margin - mean_intensity, min=0.0)
        regularization = - self.mean_intensity_w * mean_intensity
        return loss_contrastive, regularization
    
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(
            output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

class cosLoss(torch.nn.Module):
    def __init__(self):
        super(cosLoss, self).__init__()
    def forward(self, output1, output2, label):
        output1 = torch.nn.functional.normalize(output1, p=2.0, dim=-1, eps=1e-12, out=None)
        output2 = torch.nn.functional.normalize(output2, p=2.0, dim=-1, eps=1e-12, out=None)
        cos = torch.sum(output1 * output2, dim = -1, keepdim=True)
        loss = torch.sum(label * cos + (1 - label) * (1 - cos))
        return loss