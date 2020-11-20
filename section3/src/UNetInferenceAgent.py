"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask     
        """
        patch_size = 64
        volume = med_reshape(volume, (volume.shape[0], patch_size, patch_size))
        mask = np.zeros(volume.shape)
        for i in range(0, volume.shape[0]):
            ind_slice = torch.from_numpy(volume[i, : , : ].astype(np.single)).unsqueeze(0).unsqueeze(0)
            pred = self.model(ind_slice.to(self.device))
            mask[i,:, :] = torch.argmax(np.squeeze(pred.cpu().detach()), dim = 0)
        
        return mask

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        
        slices = []

        
        mask = np.zeros(volume.shape)
        for i in range(0, volume.shape[0]):
            ind_slice = torch.from_numpy(volume[i, : , : ].astype(np.single)).unsqueeze(0).unsqueeze(0)
            pred = self.model(ind_slice.to(self.device))
            mask[i,:, :] = torch.argmax(np.squeeze(pred.cpu().detach()), dim = 0)
         

        return mask 
