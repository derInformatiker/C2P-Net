import os
import pickle
from glob import glob
from fnmatch import fnmatch

from attr import has
from model.pointnet import PointNetCls, MultiModalPointNetRegistration, PointNetRegistration
from model.pointnet2.pointnet2_cls import PointNet2Cls
from model.pointnet2.pointnet2_seg import PointNet2Seg
from model.pct_seg import Pct as Pct_Seg
from sklearn.model_selection import train_test_split

from model.transmorph import TransMorph


def makeModel(args):
    if args.mode == 'classification':
        if args.model_arch == 'PointNet':
            return PointNetCls(
                        k=args.num_classes,
                        inp_dim=args.input_chan,
                        feat_trans=args.feat_trans,
                        global_size=args.global_feature_size,
                        aggregate_func=args.aggregate_func
                    )
        
        
        elif args.model_arch == 'Pct':
            from model.pct import Pct
            return Pct(args)

        elif args.model_arch == 'PointNet++':
            return PointNet2Cls(
                args.num_classes,
                args.input_chan - 3,
            )
        elif args.model_arch == 'SwinTransformer':
            from model.swintransformer import SwinTransformer, swin_config
            return SwinTransformer(**swin_config)
    elif args.mode == 'registration':
        if args.model_arch == 'PointNet':
            return PointNetRegistration(
                        k=args.num_classes,
                        inp_dim=args.input_chan,
                        feat_trans=args.feat_trans,
                        global_size=args.global_feature_size,
                        aggregate_func=args.aggregate_func
                    )
        elif args.model_arch == 'MMPointNet':
            return MultiModalPointNetRegistration(
                        k=args.num_classes,
                        inp_dim=args.input_chan,
                        feat_trans=args.feat_trans,
                        global_size=args.global_feature_size,
                        aggregate_func=args.aggregate_func
                    )
        elif args.model_arch == 'PointNet++':
            return PointNet2Seg(
                args.num_classes,
                3,
            )
        elif args.model_arch == 'Pct':
            return Pct_Seg(args)

        elif args.model_arch == 'TransMorph':
            import ml_collections
            config = ml_collections.ConfigDict()
            config.if_transskip = True
            config.if_convskip = True
            config.patch_size = 4
            config.in_chans = args.input_chan * 2
            config.embed_dim = 96
            config.depths = (2, 2, 4, 2)
            config.num_heads = (4, 4, 8, 8)
            config.window_size = 32
            config.mlp_ratio = 4
            config.pat_merg_rf = 4
            config.qkv_bias = False
            config.drop_rate = 0
            config.drop_path_rate = 0.3
            config.ape = False
            config.spe = False
            config.rpe = True
            config.patch_norm = True
            config.use_checkpoint = False
            config.out_indices = (0, 1, 2, 3)
            config.reg_head_chan = 16
            config.maxNPoints = args.maxNPoints

            return TransMorph(config)


def generateSplit(X, train_size: float):
    trainPaths, valPaths = train_test_split(
        X, train_size=train_size, random_state=0)
    return trainPaths, valPaths

def getModelNetPaths(args):
    paths = glob(
        os.path.join(args.ds_path, '*/train/*.OFF').replace('\\', '/')
    )
    testPaths = glob(
        os.path.join(args.ds_path, '*/test/*.OFF').replace('\\', '/')
    )
    return paths, testPaths

def getMeshDataPaths(args):
    paths = glob(
        os.path.join(args.ds_path, f'*/*/*.stl').replace('\\', '/')
    )
    testPaths = []
    return paths, testPaths

def makeDataset(args):
    name = args.dataset_name
    if 'shape' in name:
        from data.dataset import ShapeRegistration
        args.shape_select = name.split('_')[1]
        return ShapeRegistration
    elif name == 'ear':
        from data.dataset import EarRegistration
        args.noisy_intra = False
        return EarRegistration
    elif name == 'ear_noisy_intra':
        from data.dataset import EarRegistration
        args.noisy_intra = True
        return EarRegistration
    else:
        print('Dataset not found')