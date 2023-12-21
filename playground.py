import os
import sys
import json
import open3d as o3d
import numpy as np
import argparse
from pathlib import Path
from typing import NamedTuple
from plyfile import PlyData, PlyElement
from PIL import Image
from utils.sh_utils import SH2RGB
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
# from scene.gaussian_model import BasicPointCloud
from visualzie_pcds import visualize_pcd, visualize_multiple_pcds


def load_ply(ply_path, opacity_threshold):
    plydata = PlyData.read(ply_path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    colors = SH2RGB(features_dc)
    colors = np.reshape(colors, (xyz.shape[0], 3))


    if opacity_threshold:
        opacities = np.reshape(opacities, (xyz.shape[0]))
        condition = opacities > 5
        indices = np.where(condition)
        xyz = xyz[indices]
        colors = colors[indices]


    points = o3d.utility.Vector3dVector(xyz)
    colors2 = o3d.utility.Vector3dVector(colors)

    pcd = o3d.geometry.PointCloud(points=points)
    pcd.colors = colors2


    return pcd
def comp_dist(pcd_source, pcd_target):
    dists = pcd_source.compute_point_cloud_distance(pcd_target)
    dists = np.asarray(dists)
    # dist_length = np.linalg.norm(dists)
    return dists

if __name__ == '__main__':
    ply_1 = '/home/roosbot/gaussian-splatting/inspection_7492/model_1_v1.pcd'
    ply_2 = '/home/roosbot/gaussian-splatting/inspection_7491/pc_12.ply'

    pcd1 = o3d.io.read_point_cloud(ply_1)
    pcd2 = load_ply(ply_2, opacity_threshold=False)
    # pcd1.paint_uniform_color([0, 0, 1])
    # pcd2.paint_uniform_color([0.5, 0.5, 0])
    # visualize_multiple_pcds([pcd1, pcd2])

    dists = comp_dist(pcd2, pcd1)
    norm_dist = np.linalg.norm(dists)
    print(norm_dist)
    ind = np.where(dists < 0.1)[0]
    pcd_without_floaters = pcd2.select_by_index(ind)
    # visualize_multiple_pcds([pcd1, pcd_without_floaters])
    visualize_pcd(pcd_without_floaters)