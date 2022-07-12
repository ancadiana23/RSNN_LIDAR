# This import registers the 3D projection, but is otherwise unused.
import argparse

import matplotlib.image as mpimg
import math
import matplotlib.pyplot as plt
import numpy as np
import re
import torch
from PIL import Image
import random
import matplotlib.pyplot as plt


######################### Load functions #########################

def load_kitti_image(example_name, data_path="../data/KITTI/training/"):
    image_file = data_path + "image_2/" + example_name + ".png"
    img = mpimg.imread(image_file)
    return img


def load_kitti_point_cloud(example_name, data_path="../data/KITTI/training/"):
    velodyne_file = data_path + "velodyne/" + example_name + ".bin"
    scan = np.fromfile(velodyne_file, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan


def load_kitti_calibration(example_name, data_path="../data/KITTI/training/"):
    calib_file = data_path + "calib/" + example_name + ".txt"
    with open(calib_file) as f:
        text = f.read()
    matches = re.findall('(.*):(.*)', text)
    calibration = {key: np.array(np.matrix(value).reshape(3, -1))
                   for (key, value) in matches}
    calibration['R0_rect'] = np.append(
        calibration['R0_rect'], [[0], [0], [0]], axis=1)
    calibration['R0_rect'] = np.append(
        calibration['R0_rect'], [[0, 0, 0, 1]], axis=0)
    calibration['Tr_velo_to_cam'] = np.append(
        calibration['Tr_velo_to_cam'], [[0, 0, 0, 1]], axis=0)
    calibration['Tr_imu_to_velo'] = np.append(
        calibration['Tr_imu_to_velo'], [[0, 0, 0, 1]], axis=0)
    return calibration


def load_kitti_label(example_name, data_path="../data/KITTI/training/"):
    label_file = data_path + "label_2/" + example_name + ".txt"
    with open(label_file) as f:
        text = f.read()
    labels = [line.split() for line in text.splitlines()]
    dict_labels = []
    for label in labels:
        label[1:] = [float(item) for item in label[1:]]
        label_dict = {"type": label[0], "truncated": label[1],
                      "occluded": label[2], "alpha": label[3],
                      "bbox": (label[4], label[5], label[6], label[7]),
                      "dimensions": (label[8], label[9], label[10]),
                      "location": (label[11], label[12], label[13]),
                      "rotation_y": label[14]}
        dict_labels.append(label_dict)
    return dict_labels


def get_all_labels():
    return ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram"]

######################### Post processing #########################

# dataset: list of (image, target) tuples
def normalize_examples(dataset):
    max_val = 0.0
    for i in range(len(dataset)):
        local_max = dataset[i][0].max()
        if local_max > max_val:
            max_val = local_max
    dataset = [(data / max_val, target) for (data, target) in dataset]
    return dataset

# dataset: list of (image, target) tuples
def depth_clip(dataset, max_val=150.0):
    for i in range(len(dataset)):
        dataset[i][0][dataset[i][0] > max_val] = max_val
    return dataset

######################### Object Projection and selection #########################

def extract_object_from_image(example_name, label_index, object_size=(300, 500)):
    labels = load_kitti_label(example_name)
    image = load_kitti_image(example_name)
    bbox = tuple(map(int, labels[label_index]["bbox"]))
    object = np.zeros((object_size[0], object_size[1], 3))

    label_height = bbox[3] - bbox[1]
    label_width = bbox[2] - bbox[0]
    (obj_height, obj_width) = object_size

    margin_height = round((obj_height - label_height) / 2.0)
    margin_width = round((obj_width - label_width) / 2.0)

    img_top = bbox[1] - margin_height
    img_bottom = img_top + obj_height
    img_left = bbox[0] - margin_width
    img_right = img_left + obj_width

    obj_top = 0
    obj_bottom = obj_height
    obj_left = 0
    obj_right = obj_width
    if img_top < 0:
        obj_top = -img_top
        img_top = 0
    if img_bottom >= image.shape[0]:
        obj_bottom = obj_bottom - (img_bottom - image.shape[0])
        img_bottom = image.shape[0]
    if img_left < 0:
        obj_left = -img_left
        img_left = 0
    if img_right >= image.shape[1]:
        obj_right = obj_right - (img_right - image.shape[1])
        img_right = image.shape[1]
    object[obj_top:obj_bottom,
           obj_left:obj_right] = image[img_top:img_bottom, img_left:img_right]
    return object


def project_3D_coordinates(point_cloud, calibration, velodyne_coordinates=True):
    points_3D = point_cloud[:, :3]
    points_3D = np.append(points_3D, np.ones(
        (point_cloud.shape[0], 1)), axis=1)
    points_3D = points_3D.T
    if velodyne_coordinates:
        # project to camera coordinates
        points_3D = calibration['R0_rect'] @ calibration['Tr_velo_to_cam'] @ points_3D
    points_2D = calibration['P2'] @ points_3D

    points_2D[0, :] = points_2D[0, :] / points_2D[2, :]
    points_2D[1, :] = points_2D[1, :] / points_2D[2, :]
    return points_2D


def project_single_objects(point_cloud, calibration, labels, img_size,
                           object_size, object_ratio, scale_factor, plot=False,
                           image=None, save_path=None):
    points_2D = project_3D_coordinates(point_cloud, calibration)

    all_labels = get_all_labels()
    (height, width) = img_size
    img = np.zeros((int((height / scale_factor) + 1),
                   int((width / scale_factor) + 1)))
    objects = []
    label_indices = []
    for label_idx, label in enumerate(labels):
        new_bbox = get_object_margins(label["bbox"], object_size, object_ratio)
        if new_bbox is None or label["type"] in ['Misc', 'DontCare']:
            continue

        (left, top, right, bottom) = new_bbox
        # select the indices of the points contained in the bounding box
        indices = (points_2D[0] >= left) & (points_2D[0] < right) & (
            points_2D[1] >= top) & (points_2D[1] < bottom)
        values = points_2D[2, indices]

        xs = ((points_2D[1, indices] - top) / scale_factor).astype(int)
        ys = ((points_2D[0, indices] - left) / scale_factor).astype(int)

        obj_img = np.zeros((1, int(object_size[0] / scale_factor + 1),
                            int(object_size[1] / scale_factor + 1)))
        for (x, y, value, i) in zip(xs, ys, values, np.arange(len(indices))[indices]):
            if value > obj_img[0, x, y]:
                obj_img[0, x, y] = value
                if points_2D[0, i] >= 0 and points_2D[0, i] < width and points_2D[1, i] >= 0 and points_2D[1, i] < height:
                    img[x + int(top / scale_factor), y +
                        int(left / scale_factor)] = value
        objects.append((obj_img, all_labels.index(label["type"])))
        label_indices.append(label_idx)

    return objects, label_indices


def get_object_margins(bbox, object_size, object_ratio):
    label_height = bbox[3] - bbox[1]
    label_width = bbox[2] - bbox[0]
    (obj_height, obj_width) = object_size

    if label_height > obj_height or label_width > obj_width:
        return None
    if label_height < obj_height / object_ratio and label_width < obj_width / object_ratio:
        return None
    margin_height = (obj_height - label_height) / 2.0
    margin_width = (obj_width - label_width) / 2.0
    return (bbox[0] - margin_width, bbox[1] - margin_height,
            bbox[2] + margin_width, bbox[3] + margin_height)


def extract_kitti_objects(params, start=0, end=5000,
                          data_path="../data/KITTI/training/",
                          save_path="../data/processed/",
                          save_name="training"):
    objects = []
    image_label_pointers = []
    for i in range(start, end):
        example_name = "%06d" % i
        point_cloud = load_kitti_point_cloud(example_name, data_path=data_path)
        original_image = load_kitti_image(example_name, data_path=data_path)
        calibration = load_kitti_calibration(example_name, data_path=data_path)
        labels = load_kitti_label(example_name, data_path=data_path)
        save_plot_path = save_path + "object_plots/" + example_name + ".eps"
        img_objects, label_indices = project_single_objects(
            point_cloud, calibration, labels,
            original_image.shape[:2], params.object_size,
            params.object_ratio, params.scale_factor,
            plot=True, image=original_image,
            save_path=save_plot_path)
        objects += img_objects
        image_label_pointers += [(i, label) for label in label_indices]

    objects = depth_clip(objects)
    objects = normalize_examples(objects)
    torch.save([objects, image_label_pointers, params], save_path + save_name)


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_size', type=int, default=[232, 392], nargs="+",
                        help='Dimention of the windows around objects.')
    parser.add_argument('--object_ratio', type=float, default=5.0,
                        help='Minimum size of object; e.g. if object ratio is 5, the object has to be cover at least a fifth of the cropped image.')
    parser.add_argument('--scale_factor', type=float, default=8.0,
                        help='The final object images are resized by this factor.')

    FLAGS, unparsed = parser.parse_known_args()

    dataset_name = "processed/"
    extract_kitti_objects(FLAGS, start=0, end=50,
                          data_path="../data/KITTI/training/",
                          save_path="../data/" + dataset_name,
                          save_name="training.pt")
    extract_kitti_objects(FLAGS, start=50, end=74,
                          data_path="../data/KITTI/training/",
                          save_path="../data/" + dataset_name,
                          save_name="testing.pt")
