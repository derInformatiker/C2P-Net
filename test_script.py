# %%
# IMPORTS FOR NGENET AND DEFORMATION PYRAMID
import os
from glob import glob
from pickle import load, dump
import argparse

import numpy as np
import open3d as o3d
import torch
import trimesh as trm
import yaml
import time
from scipy.spatial.distance import cdist
from statistics import mean
from easydict import EasyDict as edict

from deformationpyramid.model.geometry import *
from deformationpyramid.model.loss import compute_truncated_chamfer_distance
from deformationpyramid.model.registration import Registration
from tqdm import tqdm
from ngenet.data import EarDataset, EarDatasetTest, get_dataloader
from ngenet.models import NgeNet, architectures, vote
from ngenet.utils import (decode_config, execute_global_registration, get_blue,
                          get_correspondences, get_yellow, npy2feat, npy2pcd, pcd2npy,
                          setup_seed, to_tensor, vis_plys)
from deformationpyramid.utils.benchmark_utils import setup_seed
from deformationpyramid.utils.tiktok import Timers

# %%
def join(loader, node):
    seq = loader.construct_sequence(node)
    return '_'.join([str(i) for i in seq])
yaml.add_constructor('!join', join)

def checkIPython():
    try:
        get_ipython().__class__.__name__
        return True
    except:
        return False

if checkIPython(): # Checks if running in IPython notebook. If running by CLI, argparse is used
    class config:
        pass
    args = config()
    args.data_root = 'mesh_dataset/ear_dataset/'
    args.dataset_split = 'test'
    args.oct_data_root = 'mesh_dataset/oct_outputs/*.npy'
    args.checkpoint = 'trainResults/eardataset_nonrigid_randrot_pretrained_eardrum_large_ds/checkpoints/best_recall.pth'
    #args.checkpoint = 'trainResults/trained_03_13_100k_randomsplit_0/checkpoints/best_recall.pth'
    args.vis = False
    args.no_cuda = False
    args.use_real = False

    args.ngenet_config_path = 'config/eardataset_ngenet.yaml'
    args.ndp_config_path = 'config/NDP.yaml'
else:
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='mesh_dataset/ear_dataset/', help='root of synthetic dataset')
    parser.add_argument('--dataset_split', type=str, default='val', help='which of the splits should be used as synthetic dataset')
    parser.add_argument('--oct_data_root', type=str, default='mesh_dataset/oct_outputs/*.npy', help='glob specification of all oct scans to test (convert to .npy first)')
    parser.add_argument('--checkpoint', type=str, default='trainResults/eardataset_nonrigid_randrot_pretrained_eardrum_large_ds/checkpoints/best_loss.pth', help='path to NgeNet checkpoint')
    parser.add_argument('--ngenet_config_path', type=str, default='config/eardataset.yaml', help='which configuration file to use for NgeNet')
    parser.add_argument('--ndp_config_path', type=str, default='config/NDP.yaml', help='which configuration file to use for NDP')
    parser.add_argument('--use_real', action='store_true', default=False, help='decide wheather to run the test on the oct scans')
    parser.add_argument('--vis', action='store_true',  default=False, help='visualize output while running in an extra window')
    parser.add_argument('--no_cuda', action='store_true',  default=False, help='disable cuda (cpu only)')

    args = parser.parse_args()


# %%
setup_seed(22)
config = decode_config(args.ngenet_config_path)
config = edict(config)
config.architecture = architectures[config.dataset]
config.num_workers = 0

all_oct_outputs = glob(args.oct_data_root)

test_dataset = EarDataset(
    root=args.data_root,
    noisy_intra=config.noisy_intra,
    split=args.dataset_split,
    aug=False,
    overlap_radius=config.overlap_radius
)

test_dataset_real = EarDatasetTest(
    test_paths=all_oct_outputs,
    root=args.data_root,
    noisy_intra=config.noisy_intra,
    split='test',
    aug=False,
    overlap_radius=config.overlap_radius
)

metadata = test_dataset.metadata

test_dataloader, neighborhood_limits = get_dataloader(
    config=config,
    dataset=test_dataset_real if args.use_real else test_dataset,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    shuffle=False,
    neighborhood_limits=None
)

with open(args.ndp_config_path,'r') as f:
    p_config = yaml.load(f, Loader=yaml.Loader)

p_config = edict(p_config)

# %%
with open('mesh_dataset/landmarks/landmarks.pkl', 'rb') as f:
    landmarks = load(f)

def registration_cd(points_pred, points_tgt, corrs, pred_T=np.eye(4)):
    '''

    :param points_pred: (n, 3)
    :param points_tgt: (m, 3)
    :return: float
    '''
    if len(corrs) != 0:
        R, t = pred_T[:3, :3], pred_T[:3, 3]
        points_pred = points_pred @ R.T + t
        points_pred = torch.tensor(points_pred[corrs]).T.unsqueeze(0).float()
        points_tgt = torch.tensor(points_tgt).T.unsqueeze(0).float()
        cd = compute_truncated_chamfer_distance(points_pred, points_tgt, trunc=1e+9)
        cd = cd.item()
    else:
        cd = 1e+4
    return cd

def mean_displacement_error(dis_pred, dis_gt):
    return np.linalg.norm(dis_pred-dis_gt, axis=1).mean()

def denorm(arr, metadata):
    return arr * metadata['std'] + metadata['mean']

def landmark_loss(pred, intra):
    assert len(pred) == len(intra), 'len(pred) != len(intra)'
    l = []
    for seg in range(len(pred)):
        mat = cdist(pred[seg], intra[seg]).min(0)
        
        l.append(mat.mean())
    return sum(l)/len(l)

# %%
model_rigid = NgeNet(config)
use_cuda = not args.no_cuda
if use_cuda:
    model = model_rigid.cuda()
    if args.checkpoint != None:
        model_rigid.load_state_dict(torch.load(args.checkpoint))
    p_config.device = torch.cuda.current_device()
else:
    model_rigid.load_state_dict(
        torch.load(args.checkpoint, map_location=torch.device('cpu')))
    config.device = torch.device('cpu')
model_rigid.eval()

fmr_threshold = 0.05
rmse_threshold = 0.2
inlier_ratios, mutual_inlier_ratios = [], []
mutual_feature_match_recalls, feature_match_recalls = [], []
transformations = []
nonregistered_cd_l, registered_cd_l, mean_displacement_error_l, landmark_loss_l, ngenet_cd_l = [], [], [], [], []
overlap_scores, wall_time_models = [], []
displ_ngenet, displ = [], []

dist_thresh_maps = {
    '10000': config.first_subsampling_dl,
    '5000': config.first_subsampling_dl,
    '2500': config.first_subsampling_dl * 1.5,
    '1000': config.first_subsampling_dl * 1.5,
    '500': config.first_subsampling_dl * 1.5,
    '250': config.first_subsampling_dl * 2,
}
model_nonrigid = Registration(p_config)
timer = Timers()

# %%
for pair_ind, inputs in enumerate(tqdm(test_dataloader)):
    t1 = time.time()
    if use_cuda:
        for k, v in inputs.items():
            if isinstance(v, list):
                for i in range(len(v)):
                    inputs[k][i] = inputs[k][i].cuda()
            else:
                inputs[k] = inputs[k].cuda()
    with torch.no_grad():
        
        batched_feats_h, batched_feats_m, batched_feats_l = model_rigid(inputs)
        stack_points = inputs['points']
        stack_points_raw = inputs['batched_points_raw']
        stack_lengths = inputs['stacked_lengths']
        coords_tgt_full = inputs['points_tgt_full']
        coords_src = stack_points[0][:stack_lengths[0][0]]
        coords_tgt = stack_points[0][stack_lengths[0][0]:]
        coords_src_raw = stack_points_raw[:stack_lengths[0][0]]
        coords_tgt_raw = stack_points_raw[stack_lengths[0][0]:]
        feats_src_h = batched_feats_h[:stack_lengths[0][0]]
        feats_tgt_h = batched_feats_h[stack_lengths[0][0]:]
        feats_src_m = batched_feats_m[:stack_lengths[0][0]]
        feats_tgt_m = batched_feats_m[stack_lengths[0][0]:]
        feats_src_l = batched_feats_l[:stack_lengths[0][0]]
        feats_tgt_l = batched_feats_l[stack_lengths[0][0]:]

        displacement_gt = inputs['transf'][0].detach().cpu().numpy() * metadata['std']

        source_npy = coords_src.detach().cpu().numpy()
        target_npy = coords_tgt.detach().cpu().numpy()

        source_feats_h = feats_src_h[:, :-2].detach().cpu().numpy()
        target_feats_h = feats_tgt_h[:, :-2].detach().cpu().numpy()
        source_feats_m = feats_src_m.detach().cpu().numpy()
        target_feats_m = feats_tgt_m.detach().cpu().numpy()
        source_feats_l = feats_src_l.detach().cpu().numpy()
        target_feats_l = feats_tgt_l.detach().cpu().numpy() 
        
        after_vote = vote(
            source_npy=source_npy, 
            target_npy=target_npy, 
            source_feats=[source_feats_h, source_feats_m, source_feats_l], 
            target_feats=[target_feats_h, target_feats_m, target_feats_l], 
            voxel_size=config.first_subsampling_dl,
            use_cuda=use_cuda)
        source_npy, target_npy, source_feats_npy, target_feats_npy = after_vote

        source, target = npy2pcd(source_npy), npy2pcd(target_npy)
        
        source_feats, target_feats = npy2feat(source_feats_h), npy2feat(target_feats_h)
        pred_T, estimate, result = execute_global_registration(
            source=source,
            target=target,
            source_feats=source_feats,
            target_feats=target_feats,
            voxel_size=dist_thresh_maps['10000']
        )
        
        transformations.append(pred_T)
    
    corrs_pred = np.unique(np.asarray(result.correspondence_set).T[0])
    
    raw_coords_src = torch.tensor(denorm(pcd2npy(estimate), metadata)).to(coords_src.device)
    raw_coords_tgt = torch.tensor(denorm(pcd2npy(target), metadata)).to(coords_tgt.device)

    coords_src_raw = coords_src_raw.cpu().detach().numpy()
    coords_tgt_raw = coords_tgt_raw.cpu().detach().numpy()
    coords_tgt_full = coords_tgt_full.cpu().detach().numpy()
    coords_transformed_src = raw_coords_src.cpu().detach().numpy()

    inds = inputs['inds'][0].cpu().detach().numpy()
    faces = inputs['faces'][0].cpu().detach().numpy()
    
    model_nonrigid.load_pcds(raw_coords_src.float(), raw_coords_tgt.float(), inds=corrs_pred, search_radius=0.0375)
    warped_pcd, hist, iter_cnt, timer = model_nonrigid.register(visualize=args.vis, timer = timer)
    warped_pcd = warped_pcd.cpu().numpy()
    
    
    t2 = time.time()
    wall_time_models.append(t2-t1)
    
    displacement_ngenet = raw_coords_src.cpu().detach().numpy() - coords_src_raw
    displacement_pred = warped_pcd - coords_src_raw
    displ_ngenet.append(displacement_ngenet)
    displ.append(displacement_pred)
    
    if not args.use_real:
        registered_cd = compute_truncated_chamfer_distance(
            torch.tensor(warped_pcd).unsqueeze(0), 
            torch.tensor(coords_tgt_full).unsqueeze(0), 
            trunc=1e+9
        ).item()
        
        nonregistered_cd = compute_truncated_chamfer_distance(
            torch.tensor(coords_src_raw).unsqueeze(0), 
            torch.tensor(coords_tgt_full).unsqueeze(0), 
            trunc=1e+9
        ).item()

        ngenet_cd = compute_truncated_chamfer_distance(
            torch.tensor(coords_transformed_src).unsqueeze(0).float(), 
            torch.tensor(coords_tgt_full).unsqueeze(0), 
            trunc=1e+9
        ).item()

        mde = mean_displacement_error(
            displacement_pred, 
            displacement_gt
        )

        l_inds = [v for u, v in landmarks.items()]
        pred_landmarks = [warped_pcd[i] for i in l_inds]
        pre_landmarks = [coords_tgt_full[i] for i in l_inds]

        lndmk = landmark_loss(pre_landmarks, pred_landmarks)
        ind = metadata[args.dataset_split][pair_ind].split("/")[1]
        landmark_loss_l.append(lndmk)
        ngenet_cd_l.append(ngenet_cd)
    else:
        
        registered_cd = compute_truncated_chamfer_distance(
            torch.tensor(warped_pcd).unsqueeze(0), 
            torch.tensor(coords_tgt_raw).unsqueeze(0), 
            trunc=1e+9
        ).item()
        
        nonregistered_cd = compute_truncated_chamfer_distance(
            torch.tensor(coords_src_raw).unsqueeze(0), 
            torch.tensor(coords_tgt_raw).unsqueeze(0), 
            trunc=1e+9
        ).item()

        ind = all_oct_outputs[pair_ind].split('\\')[-1]
        with open(f'mesh_dataset/oct_outputs/{ind.split(".")[0]}_lndmrks.pkl', 'rb') as f:
            landmarks_intra = load(f)
        
        if landmarks_intra != {}:
            pred_landmarks = [warped_pcd[landmarks[k]] for k, v in landmarks_intra.items()]
            intra_landmarks = [v for k, v in landmarks_intra.items()]
            lndmk = landmark_loss(pred_landmarks, intra_landmarks)
            landmark_loss_l.append(lndmk)
        mde = -1
    
    overlap = len(target_npy)/len(source_npy)
    overlap_scores.append(overlap)
    registered_cd_l.append(registered_cd)
    nonregistered_cd_l.append(nonregistered_cd)
    mean_displacement_error_l.append(mde)
    
    
    output_mesh = trm.Trimesh(warped_pcd, faces)
    _=output_mesh.export(f'test_output_folder/predictions/prediction_{ind}.stl')
    np.save(f'test_output_folder/pred_corrs/pred_corrs_{ind}.npy',np.asarray(result.correspondence_set))
    np.save(f'test_output_folder/ndp_hist/ndp_hist_{ind}.npy',hist)

# %%
print('Non-registered cd score:', mean(nonregistered_cd_l))
print('Registered cd score:', mean(registered_cd_l))
print('Overlap of pointclouds:', mean(overlap_scores))
print('Mean displacement error: ', mean(mean_displacement_error_l))
print('Landmark loss:', mean(landmark_loss_l))
print('Wall time:', mean(wall_time_models), 's')

# %%
vis_plys([estimate, target])

# %%
""" Non-registered cd score: 618.4198693275451
Registered cd score: 1.993619680404663
Overlap of pointclouds: 1.0
Mean displacement error:  -1
Landmark loss: 0.7676651613473052
Wall time: 8.181469106674195 s """

# %%



