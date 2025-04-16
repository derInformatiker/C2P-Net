import os
import pickle

import mesh_dataset.vtkutils as vtkutils
import numpy as np
import vtk
import glob
import trimesh as trm
from scipy.spatial.transform import Rotation
from sklearn.neighbors import KDTree
from torch.utils.data import Dataset
from vtk.util.numpy_support import vtk_to_numpy

CUR = os.path.dirname(os.path.abspath(__file__))
from ngenet.utils import get_correspondences, normal, npy2pcd


def artifacting(vert, faces, surface_amount=0.5, random_noise=False, move_rat=0.2, move_mean=0, move_std=1):
    '''
    Radomly removes points from pointcloud by using centroids
    '''
    if random_noise:
        n_m = round(move_rat * len(vert))
        m_idx = np.random.choice(np.arange(len(vert)), size=(n_m))
        noise = np.random.normal(move_mean, move_std, (n_m, 3))
        vert[m_idx] += noise

    mesh = vtkutils.createPolyData(verts=vert, tris=np.transpose(faces, axes=(1, 0)).astype(np.int64))
    cellsArray = vtk.vtkCellArray()
    for c in enumerate(np.transpose(faces, axes=(1, 0)).astype(np.int64).T):
        cellsArray.InsertNextCell( 3, c[1] )
    mesh.SetPolys(cellsArray)

    noisy_vert = vtkutils.randomSurface(mesh, surfaceAmount=surface_amount)
    noisy_vert = vtk_to_numpy(noisy_vert.GetPoints().GetData())
    return noisy_vert

class EarDataset(Dataset):
    def __init__(self, root, noisy_intra, split, aug, overlap_radius, noise_scale=0.005, surface_amount=None):
        super().__init__()
        self.root = root
        self.split = split
        self.noisy = noisy_intra
        self.aug = aug
        self.noise_scale = noise_scale
        self.overlap_radius = overlap_radius
        self.surface_amount = surface_amount
        self.max_points = 30000
        
        with open(os.path.join(root, 'metadata.pkl'), 'rb') as f:
            self.metadata = pickle.load(f)

        self.paths = [os.path.join(root, i.split("/")[-1]) for i in self.metadata[split]]

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

        tgt_points_full = data['points_intra']
        tgt_points_raw = data['points_intra_noisy' if self.noisy else 'points_intra']
        if self.surface_amount != None:
            tgt_points_raw = artifacting(data['points_intra'], data['faces'], surface_amount=self.surface_amount)
        tgt_points = self.norm(tgt_points_raw) # npy, (m, 3)

        displ = data['displacement']/self.metadata['std']

        coors = get_correspondences(npy2pcd(src_points+displ),
                                    npy2pcd(tgt_points),
                                    np.eye(4),
                                    self.overlap_radius)

        if self.aug:
            euler_ab = np.random.rand(3) * 1 * np.pi
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            center = np.median(tgt_points_raw, axis=0)
            tgt_points_raw = ((tgt_points_raw - center) @ rot_ab.T) + center
            tgt_points = self.norm(tgt_points_raw)
            
            center = np.median(tgt_points, axis=0)
            tgt_points_full_norm = ((self.norm(tgt_points_full) - center) @ rot_ab.T) + center
            displ = (tgt_points_full_norm-src_points)

        src_feats = np.ones_like(src_points[:, :1], dtype=np.float32)
        tgt_feats = np.ones_like(tgt_points[:, :1], dtype=np.float32)

        src_pcd, tgt_pcd = normal(npy2pcd(src_points)), normal(npy2pcd(tgt_points))
        src_normals = np.array(src_pcd.normals).astype(np.float32) 
        tgt_normals = np.array(tgt_pcd.normals).astype(np.float32)

        if self.surface_amount != None:
            tree = KDTree(data['points_intra'])
            distances, indices = tree.query(tgt_points_raw, 1)
            inds = indices[:, 0]
        else:
            inds = data['intra_inds']

        
        pair = dict(
            src_points=src_points,
            src_faces=src_faces,
            tgt_points=tgt_points,
            src_feats=src_feats,
            tgt_feats=tgt_feats,
            tgt_points_full=tgt_points_full,
            src_normals=src_normals,
            tgt_normals=tgt_normals,
            transf=displ,
            coors=coors,
            inds=inds,
            src_points_raw=src_points_raw,
            tgt_points_raw=tgt_points_raw,
            faces=data['faces'])
        return pair

class EarDatasetTest(Dataset):
    def __init__(self, test_path, root, noisy_intra, split, aug, overlap_radius, noise_scale=0.005):
        super().__init__()
        self.root = root
        self.split = split
        self.noisy = noisy_intra
        self.aug = aug
        self.noise_scale = noise_scale
        self.overlap_radius = overlap_radius
        self.max_points = 30000
        self.test_paths = glob.glob(os.path.join(test_path, 'sample_*'))
        
        with open(os.path.join(test_path, 'metadata.pkl'), 'rb') as f:
            self.metadata = pickle.load(f)

        with open(os.path.join(root, 'metadata.pkl'), 'rb') as f:
            self.eardataset_metadata = pickle.load(f)

        self.paths = [os.path.join(root, i.split("/")[-1]) for i in self.eardataset_metadata[split]]
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

        #with open(f'{path}/regtr_pred.pkl', 'rb') as f:
        #    src_points_raw = pickle.load(f)['src']

        src_points_raw = data['points_pre'] # preoperative model is always the same
        src_points, src_faces = self.norm(src_points_raw, self.eardataset_metadata), data['faces'] # npy, (n, 3); npy, (f, 3)

        tgt_points_full = np.array([])
        tgt_points_raw = test_data['target_xyz']
        tgt_points = self.norm(tgt_points_raw, self.eardataset_metadata) # npy, (m, 3)

        displ = np.array([])

        coors = np.array([])

        src_feats = np.ones_like(src_points[:, :1], dtype=np.float32)
        tgt_feats = np.ones_like(tgt_points[:, :1], dtype=np.float32)

        src_pcd, tgt_pcd = normal(npy2pcd(src_points)), normal(npy2pcd(tgt_points))
        src_normals = np.array(src_pcd.normals).astype(np.float32) 
        tgt_normals = np.array(tgt_pcd.normals).astype(np.float32)

        inds = np.array([])


        pair = dict(
            src_points=src_points,
            src_faces=src_faces,
            tgt_points=tgt_points,
            src_feats=src_feats,
            tgt_feats=tgt_feats,
            tgt_points_full=tgt_points_full,
            src_normals=src_normals,
            tgt_normals=tgt_normals,
            transf=displ,
            coors=coors,
            inds=inds,
            src_points_raw=src_points_raw,
            tgt_points_raw=tgt_points_raw,
            faces=data['faces'],
            landmarks=test_data['landmarks'],
            segmentation=test_data['segmentation'],
            metadata=test_data['metadata'])
        return pair
