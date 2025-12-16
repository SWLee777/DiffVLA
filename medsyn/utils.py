from typing import Tuple
import numpy as np
import numpy.typing as npt


def accumulate_rotation(init_matrix: npt.NDArray, angle: float, axes: Tuple[int, int]) -> npt.NDArray:

    cos_ang = np.cos(angle)
    sin_ang = np.sin(angle)

    rot_mat = np.identity(len(init_matrix))


    a1, a2 = axes
    rot_mat[a1, a1] = rot_mat[a2, a2] = cos_ang
    rot_mat[a1, a2] = -sin_ang
    rot_mat[a2, a1] = sin_ang
    return rot_mat @ init_matrix


def accumulate_scaling(init_matrix: npt.NDArray, scale: float) -> npt.NDArray:

    scale_mat = np.identity(len(init_matrix)) * scale

    return scale_mat @ init_matrix


def get_patch_slices(patch_corner: np.ndarray, patch_shape: Tuple[int]) -> Tuple[slice]:
    return tuple([slice(c, c + d) for (c, d) in zip(patch_corner, patch_shape)])



def get_patch_image_slices(patch_corner: np.ndarray, patch_shape: Tuple[int]) -> Tuple[slice]:
    return tuple([slice(None)] + list(get_patch_slices(patch_corner, patch_shape)))


def nsa_sample_dimension(lb, ub, img_d):
    gamma_lb = 0.03
    gamma_shape = 2
    gamma_scale = 0.1

    gamma_sample = (gamma_lb + np.random.gamma(gamma_shape, gamma_scale)) * img_d

    return int(np.clip(gamma_sample, lb, ub))
