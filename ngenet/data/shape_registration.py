import numpy as np
import os
import pickle
import trimesh
from glob import glob
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
CUR = os.path.dirname(os.path.abspath(__file__))
from ngenet.utils import npy2pcd, get_correspondences, normal


class ShapeDataset(Dataset):
    def __init__(self, root, shape, split, aug, overlap_radius, noise_scale=0.005):
        super().__init__()
        self.root = root
        self.split = split
        self.aug = aug
        self.noise_scale = noise_scale
        self.overlap_radius = overlap_radius
        self.max_points = 30000

        self.unit = {i: self.load_mesh(path=f'{root}/{i}.stl') for i in ['box', 'cone','cylinder','capsule']}
        self.paths = [i.replace('\\', '/') for i in glob(f'{root}/{shape}/{split}*.stl')]
        with open(f'{root}/transf.pkl', 'rb') as f:
            self.transf = pickle.load(f)

    def __len__(self):
        return len(self.paths)

    def load_mesh(self, path):
        mesh = trimesh.load(path)
        return np.array(mesh.vertices), np.array(mesh.faces)

    def __getitem__(self, item):
        path = self.paths[item]
        filename = path.split('/')[-1]
        shape = path.split('/')[-2]
        num = (filename.split('_')[1]).split('.')[0]
        src_path, tgt_path = path, path.replace('mesh_data/', 'mesh_data_registration_artifacts/')+'.npy'
        T = self.transf[shape+num]

        src_points, src_faces = self.unit[shape] # npy, (n, 3)
        tgt_points = np.load(tgt_path) # npy, (m, 3)

        # for gpu memory
        if (src_points.shape[0] > self.max_points):
            idx = np.random.permutation(src_points.shape[0])[:self.max_points]
            src_points = src_points[idx]
        if (tgt_points.shape[0] > self.max_points):
            idx = np.random.permutation(tgt_points.shape[0])[:self.max_points]
            tgt_points = tgt_points[idx]

        if self.aug:
            euler_ab = np.random.rand(3) * 2 * np.pi
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if np.random.rand() > 0.5:
                src_points = src_points @ rot_ab.T
                rot = rot @ rot_ab.T
            else:
                tgt_points = tgt_points @ rot_ab.T
                rot = rot_ab @ rot
                trans = rot_ab @ trans

            src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.noise_scale
            tgt_points += (np.random.rand(tgt_points.shape[0], 3) - 0.5) * self.noise_scale

        

        coors = get_correspondences(npy2pcd(src_points),
                                    npy2pcd(tgt_points),
                                    T,
                                    self.overlap_radius)

        src_feats = np.ones_like(src_points[:, :1], dtype=np.float32)
        tgt_feats = np.ones_like(tgt_points[:, :1], dtype=np.float32)

        src_pcd, tgt_pcd = normal(npy2pcd(src_points)), normal(npy2pcd(tgt_points))
        src_normals = np.array(src_pcd.normals).astype(np.float32) 
        tgt_normals = np.array(tgt_pcd.normals).astype(np.float32)

        pair = dict(
            src_points=src_points,
            src_faces=src_faces,
            tgt_points=tgt_points,
            src_feats=src_feats,
            tgt_feats=tgt_feats,
            src_normals=src_normals,
            tgt_normals=tgt_normals,
            transf=T,
            coors=coors,
            src_points_raw=src_points,
            tgt_points_raw=tgt_points)
        return pair
