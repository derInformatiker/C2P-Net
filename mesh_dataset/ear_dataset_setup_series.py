'''
Run this file to create cache, split and metadata for ear_dataset.

READING .VTP AND .STL FILES DURING TRAINING IS TOO SLOW
it takes around 50ms to load one sample (wall time)
'''

import sys
import random
import tqdm
from glob import glob
from pickle import dump, load
from statistics import mean as mean_fn

import numpy as np
import trimesh as trm
import vtk
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from vtk.util.numpy_support import vtk_to_numpy

import vtkutils

random_state = 0
random.seed(10)

mean, std = [], []

root_path = 'ear_dataset'

# Extract paths from dataset
# Only these with object
paths = [i.replace('\\', '/') for i in glob(f'{root_path}/??????')]

# ear_segmentation.pkl is a pickle file which contains the segmentation for each point in mean model
with open('ear_segmentation.pkl', 'rb') as f:
    segmentation = np.array(load(f))
indexes_stape = np.where([segmentation == 2])[1]

objs = [trm.load(f'segmentations/{i}.stl') for i in ['incus', 'malleus', 'membrane', 'stape']]
umbo = [-14.9783201, -11.7483826, 9.80726337]
malleus = [-15.2788639, -12.0986271, 9.47885704]
stape = [-17.125576, -15.1159573, 10.5758705]
incus = [-17.3455696, -15.1841612, 11.0382509]

pre_surface = trm.load('ear_dataset/pre_surface.stl')
points_pre = pre_surface.vertices

reader = vtk.vtkXMLPolyDataReader()

def artifacting(vert, faces, centerPoint, surfaceAmount, random_noise=False, move_rat=0.4, move_mean=0, move_std=0.2):
    '''
    Randomly removes points from pointcloud using centroids
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
    tree = KDTree(vert)
    distances, indices = tree.query(centerPoint, 1)
    indices = indices[:, 0][0]
    noisy_vert = vtkutils.randomSurface(mesh, surfaceAmount=surfaceAmount, centerPointID=indices)
    try:
        noisy_vert = vtk_to_numpy(noisy_vert.GetPoints().GetData())
    except AttributeError:
        noisy_vert = []
    return noisy_vert

def artifactSample(points_intra):
    faces = objs[0].faces
    inc = artifacting(points_intra[segmentation == 0], faces, [incus], surfaceAmount=random.randint(5, 10)/100,random_noise=True)
    inc_seg = np.zeros(len(inc))

    faces = objs[1].faces
    mal = artifacting(points_intra[segmentation == 1], faces, [malleus], surfaceAmount=random.randint(20, 30)/100,random_noise=True)
    mal_seg = np.zeros(len(mal)) + 1

    faces = objs[2].faces
    tymp = artifacting(points_intra[segmentation == 2], faces, [umbo], surfaceAmount=random.randint(75, 90)/100,random_noise=True)
    tymp_seg = np.zeros(len(tymp)) + 2

    faces = objs[3].faces
    stap = artifacting(points_intra[segmentation == 3], faces, [stape], surfaceAmount=random.randint(3, 7)/100,random_noise=True)
    stap_seg = np.zeros(len(stap)) + 3

    noisy_points_intra = np.concatenate([i for i in [inc, mal, tymp, stap] if len(i) != 0], axis=0)
    intra_segmentation = np.concatenate([i for i in [inc_seg, mal_seg, tymp_seg, stap_seg] if len(i) != 0], axis=0)

    # CALCULATE MEAN AND STD
    cat = np.concatenate((points_pre, points_intra), axis=0)
    mean.append(cat.mean())
    std.append(cat.std())

    tree = KDTree(points_intra)
    distances, indices = tree.query(noisy_points_intra, 1)
    indices = indices[:, 0]

    return noisy_points_intra, intra_segmentation, indices


for path in tqdm.tqdm(paths):
    # SET PATHS
    intra = path + '/intra_surface.stl'
    cache = path + '/data_cached.pkl'

    # READ INTRA_SURFACE.STL FILE
    mesh = trm.load(intra)
    face_pre = mesh.faces
    points_intra = mesh.vertices
    displacement = np.array(points_intra-points_pre)
    
    results = []
    for i in range(int(sys.argv[1])):
        results.append(artifactSample(points_intra))
    
    noisy_points_intra, intra_segmentation, indices = list(map(list, zip(*results)))
    # STORE IN PICKLE
    with open(cache, 'wb') as f:
        dump({
            'points_pre': points_pre,
            'points_intra': points_intra,
            'points_intra_noisy': noisy_points_intra,
            'displacement': displacement,
            'faces': face_pre,
            'intra_inds': indices,
            'intra_segmentation': intra_segmentation
        }, f)

# Make train 90%, val 5%, test 5% split

paths_train, paths_val = train_test_split(paths, train_size=0.91225, random_state=random_state)
paths_val, paths_test = train_test_split(paths_val, train_size=0.4725, random_state=random_state)

metadata = dict(
    type='ear_dataset_series',
    train=paths_train, 
    val=paths_val, 
    test=paths_test, 
    all_paths=paths,
    num_samples=len(paths),
    mean=mean_fn(mean),
    std=mean_fn(std)
)

with open(f'{root_path}/metadata.pkl', 'wb') as f:
    dump(metadata, f)
