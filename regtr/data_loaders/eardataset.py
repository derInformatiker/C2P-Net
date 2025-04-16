# Modified by Chenpan Li

import logging
import os
import glob
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation

from regtr.utils.se3_numpy import se3_init, se3_transform, se3_inv
from regtr.utils.pointcloud import compute_overlap

# RANDOMLY PERMUTE PCD
class EarDataset(Dataset):
    def __init__(self, cfg, phase, transforms=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        assert phase in ['train', 'val', 'test']

        self.root = cfg.root
        self.split = phase
        self.noisy = True
        self.aug = False
        self.overlap_radius = cfg.overlap_radius
        self.max_points = 10000
        
        with open(os.path.join(self.root, 'metadata.pkl'), 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.paths = [os.path.join(self.root, i.split("/")[-1]) for i in self.metadata[phase]]

    def __len__(self):
        return len(self.paths)

    def load_sample(self, path):
        with open(f'{path}/data_cached.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    
    def norm(self, arr):
        return (arr-self.metadata['mean'])/self.metadata['std']

    def __getitem__(self, item):
        path = self.paths[item]
        data = self.load_sample(path)

        src_points_raw = data['points_pre']
        src_points, src_faces = self.norm(src_points_raw), data['faces'] # npy, (n, 3); npy, (f, 3)

        tgt_points_full = self.norm(data['points_intra'])
        tgt_points_raw = data['points_intra_noisy' if self.noisy else 'points_intra']
        

        tgt_points = self.norm(tgt_points_raw) # npy, (m, 3)

        displ = data['displacement']/self.metadata['std']

        if self.aug:
            euler_ab = np.random.rand(3) * 1 * np.pi
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            center = np.median(tgt_points_raw, axis=0)
            tgt_points_raw = ((tgt_points_raw - center) @ rot_ab.T) + center
            tgt_points = self.norm(tgt_points_raw)
            
            center = np.median(tgt_points, axis=0)
            tgt_points_full_norm = ((self.norm(tgt_points_full) - center) @ rot_ab.T) + center
            displ = (tgt_points_full_norm-src_points)
        
        src_overlap_mask, tgt_overlap_mask, coors = compute_overlap(
            src_points+displ,
            tgt_points,
            self.overlap_radius,
        )

        
        

        pair = {
            'src_xyz': torch.from_numpy(src_points).float(),
            'tgt_xyz': torch.from_numpy(tgt_points).float(),
            'full_tgt_xyz': torch.from_numpy(tgt_points_full).float(),
            'src_overlap': torch.from_numpy(src_overlap_mask),
            'tgt_overlap': torch.from_numpy(tgt_overlap_mask),
            'correspondences': torch.from_numpy(coors),  # indices
            'pose': torch.from_numpy(displ).float(),
            'idx': item,
            'src_path': path,
            'tgt_path': path,
            'overlap_p': 1,
        }
        return pair
    
class EarDatasetTest(Dataset):
    def __init__(self, cfg, phase, transforms=None):
        super().__init__()
        self.root = cfg.root
        self.split = phase
        self.noisy = True
        self.aug = True if phase == 'train' else False
        self.overlap_radius = cfg.overlap_radius
        self.max_points = 10000
        test_path = cfg.oct_root
        self.test_paths = glob.glob(os.path.join(test_path, 'sample_*'))
        
        with open(os.path.join(test_path, 'metadata.pkl'), 'rb') as f:
            self.metadata = pickle.load(f)

        with open(os.path.join(self.root, 'metadata.pkl'), 'rb') as f:
            self.eardataset_metadata = pickle.load(f)

        self.paths = [os.path.join(self.root, i.split("/")[-1]) for i in self.eardataset_metadata[phase]]
        self.data_sample = self.load_sample(self.paths[0])

    def __len__(self):
        return len(self.test_paths)

    def load_sample(self, path):
        with open(f'{path}/data_cached.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    
    def norm(self, arr, metadata):
        return (arr-metadata['mean'])/metadata['std']

    def __getitem__(self, item):
        path = self.test_paths[item]
        data = self.data_sample
        test_data = self.load_sample(path)

        src_points_raw = data['points_pre'] # preoperative model is always the same
        src_points, src_faces = self.norm(src_points_raw, self.eardataset_metadata), data['faces'] # npy, (n, 3); npy, (f, 3)

        tgt_points_full = np.array([])
        tgt_points_raw = test_data['target_xyz']
        tgt_points = self.norm(tgt_points_raw, self.eardataset_metadata) # npy, (m, 3)

        displ = np.array([])

        coors = np.array([])

        src_overlap_mask, tgt_overlap_mask, coors = np.array([]), np.array([]), np.array([])

        pair = {
            'src_xyz': torch.from_numpy(src_points).float(),
            'tgt_xyz': torch.from_numpy(tgt_points).float(),
            'full_tgt_xyz': torch.from_numpy(tgt_points_full).float(),
            'src_overlap': torch.from_numpy(src_overlap_mask),
            'tgt_overlap': torch.from_numpy(tgt_overlap_mask),
            'correspondences': torch.from_numpy(coors),  # indices
            'pose': torch.from_numpy(displ).float(),
            'idx': item,
            'src_path': path,
            'tgt_path': path,
            'overlap_p': 1,
            'metadata': test_data['metadata'],
            'landmarks': test_data['landmarks'],
        }
        return pair