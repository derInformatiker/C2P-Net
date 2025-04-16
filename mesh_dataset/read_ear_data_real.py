"""
Reads all .stl files from segments folder of a sample
concatenation > centering > point sampling
Usage: python read_ear_data_real [PATH TO REAL EAR DATA]
[PATH TO REAL EAR DATA]/*/segments/*.stl


from glob import glob
import numpy as np
import trimesh as trm
import sys
import open3d as o3d
from pickle import load, dump

data_path = sys.argv[1]

# Load an intra sample for centering information
intra = trm.load('ear_dataset/004991/intra_surface.stl').vertices

for sample in glob(f'{data_path}/*/segments'):
    sample_name = sample.split('\\')[1]
    print(sample_name)
    points = []
    for i in glob(f'{sample}/*.stl'):
        obj = trm.load(i)
        points.extend(obj.vertices)
    points = np.unique(np.array(points), axis=0)

    index = np.random.choice(np.arange(len(points)), (5995))
    
    points_filtered = points[index]
    center = (np.median(points_filtered, axis=0) - np.median(intra,axis=0))
    points_filtered = points_filtered - center
    landmarks_intra = {
        path.replace('\\','/').split('/')[-1].split('.')[0]:np.asarray(o3d.io.read_point_cloud(path).points) - center
        for path in glob(f'ear_data_real/{sample_name}/landmarks/*.ply')
    }
    with open(f'oct_outputs/{sample_name}_lndmrks.pkl', 'wb') as f:
        dump(landmarks_intra, f)

    if 'l' in sample_name:
        points_filtered = points_filtered * [-1, 1, 1]
        if sample_name == '2020-008-l':
            points_filtered /= 20
    np.save(f'oct_outputs/{sample_name}.npy', points_filtered)

    oct_pcd = o3d.geometry.PointCloud()
    oct_pcd.points = o3d.utility.Vector3dVector(np.array(points_filtered))
    o3d.io.write_point_cloud(f'oct_outputs/{sample_name}.ply', oct_pcd) """


import os
import glob
import torch
from torch.utils.data import Dataset
import yaml
import json
import trimesh as trm
import numpy as np
from pickle import dump, load
import trimesh as trm

from scipy.spatial.transform import Rotation as R
rot_mat = R.from_euler('xyz', [230, -10, 10], degrees=True).as_matrix()

np.random.seed(0)

class OCTSample():
    def __init__(self, sample_folder) -> None:
        self.sample_folder = sample_folder
        self.idx = int(os.path.basename(sample_folder).split("_")[1])

    def load_metadata(self, metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = yaml.load(f, Loader=yaml.FullLoader)
        return metadata


    def load_landmarks(self, center, rotation, flip=False):
        landmarks_folder = os.path.join(self.sample_folder, "annotations", "merged", "landmarks")
        landmarks_filename_list = [
            "annulus.json",
            "umbo.json",
            #"short_process_of_malleus.json",
            "malleus_handle.json",
            "long_process_of_incus.json",
            #"incus.json",
            "stapes.json"
        ]

        landmark_names = [
            "anulus",
            "Umbo",
            #"short_process_of_malleus",
            "malleus handle",
            "long process of incus",
            #"incus",
            "stape"
        ]
        

        landmarks_dict = {}
        for u, f in enumerate(landmarks_filename_list):
            if os.path.exists(os.path.join(landmarks_folder, f)):
                with open(os.path.join(landmarks_folder, f), 'r') as f:
                    landmarks_json = json.load(f)
                    # landmarks_json_list.append(json.load(f))
                    landmarks_points = np.array([c["position"]  for c in landmarks_json["markups"][0]["controlPoints"]])
                    if flip:
                        landmarks_points = landmarks_points * [-1, 1, 1] 
                    landmarks_points = (rotation @ landmarks_points.T).T 
                landmarks_dict[landmark_names[u]] = landmarks_points - center

        return landmarks_dict


    def load(self, ):
        print("loading sample: ", self.sample_folder)
        # load meta data from YAML file
        self.meta = self.load_metadata(os.path.join(self.sample_folder, 'meta_{}.yaml'.format(self.idx)))
        
        # load image
        seg_files = glob.glob(os.path.join(self.sample_folder, 'seg_*.stl'))
        xyz = []
        segmentation = []
        for file in seg_files:
            seg = trm.load(file)
            if "tympanic_membrane" in file:
                index = 0
            elif "malleus" in file:
                index = 1
            elif "incus" in file:
                index = 2
            if "stapes" in file:
                index = 3
            elif "promontory" in file:
                index = 4
                continue
            xyz.append(seg.vertices)
            
            index = -1
            segmentation.append(np.zeros((seg.vertices.shape[0]))+index)

        xyz = np.concatenate(xyz, axis=0)
        segmentation = np.concatenate(segmentation, axis=0)

        # flip left ears
        if self.meta['patient_info']['side'] == 'left':
            xyz = xyz * [-1, 1, 1]
        
        intra = trm.load('ear_dataset/004991/intra_surface.stl').vertices
        # randomly choose 2048 points
        random_indices = np.arange(len(xyz))
        np.random.shuffle(random_indices)
        xyz = xyz[random_indices[:5995]]
        segmentation = segmentation[random_indices[:5995]]
        xyz = (rot_mat @ xyz.T).T
        center = (np.median(xyz, axis=0) - np.median(intra,axis=0)) - np.array([2.579854849403921, 2.579854849403921, -1.2899274247019605])
        xyz = xyz - center

        # load landmarks
        landmarks_list = self.load_landmarks(center, rotation=rot_mat, flip=self.meta['patient_info']['side'] == 'left')

        return {
            'target_xyz': xyz,
            'segmentation': segmentation,
            "landmarks": landmarks_list,
            'metadata': self.meta,
        }


class OCTDataset(Dataset):
    def __init__(self, 
                data_folder,
                num_samples=30,
            ):
        self.data_folder = data_folder
        self.num_samples = num_samples
        self.samples = []
        self.get_samples()
        print(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def get_samples(self):
        for idx in range(self.num_samples):
            sample_folder = os.path.join(self.data_folder, "sample_{}".format(idx))
            self.samples.append(OCTSample(sample_folder))

    def __getitem__(self, idx):
        sample = self.samples[idx]
        res = sample.load()

        data = {
            'target_xyz': torch.tensor(res["target_xyz"]),
            'segmentations': torch.cat([ s.unsqueeze(dim=0) for s in res["segmentations"]], dim=0),
            "segmentation_merged": torch.tensor(res["segmentation_merged"]),
            "landmarks": res["landmarks"], # re-write collate_fn to land landmarks to tensor
        }
        return data
    

means, stds = [], []
for i in range(43):
    sample = OCTSample(f'DIOME_FanShapeCorr/sample_{i}')
    sample = sample.load()
    means.append(sample['target_xyz'].mean(0))
    stds.append(sample['target_xyz'].std(0))
    with open(f'DIOME_FanShapeCorr/sample_{i}/data_cached.pkl', 'wb') as f:
        dump(sample, f)

metadata = dict(
    mean = np.stack(means).mean(0),
    std = np.stack(stds).mean()
)

with open('DIOME_FanShapeCorr/metadata.pkl', 'wb') as f:
    dump(metadata, f)