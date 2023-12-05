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

def parse_arg():
    parser = argparse.ArgumentParser(description='generate data reader')
    parser.add_argument('--general_path', help='Path to folder where transforms and pcd are', required=True, type=str)
    parser.add_argument('--transforms_file', required=False)
    parser.add_argument('--pcd_file', required=False)
    args = parser.parse_args()
    return args

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: dict
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def convert_pcd_to_ply(path, pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file)
    name = os.path.join(path, 'points3D.ply')
    ply_file = o3d.io.write_point_cloud(name, pcd)
    return


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    print(f'type of vertices: {type(vertices)}')
    print(f'vertices: {vertices}')
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    print(f'positions: {positions.size}')
    num_pts = positions.size
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        print(f'generating random colors')
        shs = np.random.random((num_pts, 3)) / 255.0
        colors = SH2RGB(shs)

    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.zeros((num_pts, 3))
    basic_pcd = {
        'points': positions,
        'colors': colors,
        'normals': normals,
    }
    return basic_pcd
def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                              image_name=image_name, width=width, height=height);
        cam_infos.append(cam_info);
        sys.stdout.write('\n')
    return cam_infos

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []
    print(f'path passed to function: {path}')

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        only_first = True
        for idx, frame in enumerate(frames):
            if only_first:
                cam_name = os.path.join(path, frame["file_path"] + extension)
                print(f'cam_name: {cam_name}')

                # NeRF 'transform_matrix' is a camera-to-world transform
                c2w = np.array(frame["transform_matrix"])
                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                c2w[:3, 1:3] *= -1

                # get the world-to-camera transform and set R, T
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]

                image_path = os.path.join(path, cam_name)
                print(f'image_path: {image_path}')
                image_name = Path(cam_name).stem
                print(f'image_name: {image_name}')
                image = Image.open(image_path)

                im_data = np.array(image.convert("RGBA"))

                bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

                norm_data = im_data / 255.0
                arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy
                FovX = fovx

                cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                            image_path=image_path, image_name=image_name, width=image.size[0],
                                            height=image.size[1]))

            else:
                break
            only_first = False

    return cam_infos


def readMainbladesCameras(path, transformsfile, white_background):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        focal_length_x = contents["fl_x"]
        focal_length_y = contents["fl_y"]
        width = contents["w"]
        height = contents["h"]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        frames = contents["frames"]

        for idx, frame in enumerate(frames):
            # print(f'in first frame')

            cam_name = os.path.join(path, frame["file_path"])

            # NeRF 'transform_matrix' is a camera-to-world transform
            # print(f'cam_name: {cam_name}')
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            # print(f'image path: {image_path}')
            image = Image.open(image_path)
            image = image.resize((500, 500))
            # print(f'image opened')
            im_data = np.array(image.convert("RGBA"))
            # print(f'done converting RGBA data')
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[0],
                                        height=image.size[1]))

    return cam_infos

def readInspectionInfo(path, white_background, eval, images, llffhold=8):
    # `1. load cameras
    # 2 load points (if available otherwise randomize, or even a solution in between = e.g. chop up map point cloud based on camera frustums)
    # 3 whatever else is needed like NORMALIZATION
    # 4:
    reading_dir = 'images' if images == None else images
    print(f'readMainbladesCameras')
    inspection_cam_infos = readMainbladesCameras(path, "transforms_inspection.json", white_background)
    print(f'done reading cameras')

    if eval:
        train_cam_infos = [c for idx, c in enumerate(inspection_cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(inspection_cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = inspection_cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)


    ply_path = os.path.join(path, 'points3D.ply')
    try:
        pcd = fetchPly(ply_path)
        print(f'ply_path fetched')
    except:
        print(f'FAILED FETCHING PLY')
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

if __name__ == '__main__':
    args = parse_arg()
    path = args.general_path
    eval = False

    if args.transforms_file == None:
        transformsfile = path + 'transforms.json'
    else:
        transforms_file = args.transforms_file

    if args.pcd_file == None:
        pcd_file = path + 'points3D.pcd'
    else:
        pcd_file = args.pcd_file

    images = path + 'images'
    # cam_infos = readCamerasFromTransforms(path, transformsfile, False, extension=".png")
    scene_info = readInspectionInfo(path, False, eval, images)

    print(scene_info)