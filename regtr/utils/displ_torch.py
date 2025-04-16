from typing import List, Union
from scipy.interpolate import NearestNDInterpolator
import torch
from torch import Tensor

def displ_transform_list(displ: Union[List[Tensor], Tensor], xyz: List[Tensor]):
    """Similar to se3_transform, but processes lists of tensors instead

    Args:
        pose: List of (N, 3)
        xyz: List of (N, 3)

    Returns:
        List of transformed xyz
    """

    B = len(xyz)

    transformed_all = []
    for b in range(B):
        transformed = xyz[b] + displ[b]
        transformed_all.append(transformed)

    return transformed_all

def displ_interp_transform_list(displ: Union[List[Tensor], Tensor], src_xyz:List[Tensor], xyz: List[Tensor]): ### INTERP could have large impact
    """Similar to se3_transform, but processes lists of tensors instead

    Args:
        pose: List of (N, 3)
        src_xyz List of (N, 3)
        xyz: List of (M, 3)

    Returns:
        List of transformed xyz
    """

    B = len(xyz)

    transformed_all = []
    for b in range(B):
        interp = NearestNDInterpolator(src_xyz[b].cpu().numpy(), displ[b].cpu().numpy())
        transformed = torch.from_numpy(interp(xyz[b].cpu().numpy())).cuda() + xyz[b]
        transformed_all.append(transformed)

    return transformed_all

def displ_interp_transform_list_inv(displ: Union[List[Tensor], Tensor], src_xyz:List[Tensor], xyz: List[Tensor]):
    """Similar to se3_transform, but processes lists of tensors instead

    Args:
        pose: List of (N, 3)
        src_xyz List of (N, 3)
        xyz: List of (M, 3)

    Returns:
        List of transformed xyz
    """

    B = len(xyz)

    transformed_all = []
    for b in range(B):
        interp = NearestNDInterpolator((src_xyz[b] + displ[b]).cpu().numpy(), (displ[b]*(-1)).cpu().numpy())
        transformed = torch.from_numpy(interp(xyz[b].cpu().numpy())).cuda() + xyz[b]
        transformed_all.append(transformed)

    return transformed_all