import argparse

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import kitti_utils


def get_dataset_sampler(dataset, num_classes=7):
    targets = torch.tensor([float(target) for (_, target) in dataset])
    weight_per_class = torch.histc(targets, bins=num_classes, min=0, max=6)
    weight_per_class = 1 / (weight_per_class + 1e-3)
    sample_weights = torch.tensor(
        [weight_per_class[int(target)] for target in targets])
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights.type('torch.DoubleTensor'), len(dataset))
    return sampler


def resample_dataset(dataset):
    sampler = get_dataset_sampler(dataset)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=len(dataset),
                                              sampler=sampler)
    new_dataset = next(iter(data_loader))
    new_dataset = list(zip(new_dataset[0], [int(x) for x in new_dataset[1]]))
    return new_dataset


def group_indices_by_label(dataset, labels):
    indices_by_label = {label: [] for label in labels}
    for i, (_, label) in enumerate(dataset):
        indices_by_label[label].append(i)
    return indices_by_label


def plot_value_histogram(dataset, dataset_title="Training", bins=100):
    examples = torch.stack([torch.tensor(example) for (example, _) in dataset])
    example_array = np.array(examples.reshape(-1))
    plt.hist(example_array[example_array != 0.0],
             label=dataset_title, bins=bins)
    plt.ylim(0, 1000000)
    plt.legend()


def analyse_kitti_objects_dataset(dataset_path, plot_images=True, resample=False, result_path=""):
    train_data = torch.load(dataset_path + "training.pt")
    test_data = torch.load(dataset_path + "testing.pt")
    result_path = dataset_path + "analysis/"

    train_dataset = train_data[0]
    test_dataset = test_data[0]
    params = train_data[2]

    resampled_train_dataset = resample_dataset(train_dataset)
    resampled_test_dataset = resample_dataset(test_dataset)
    label_names = kitti_utils.get_all_labels()

    plt.title("Image value histogram")
    bins = 100
    plt.figure()
    plot_value_histogram(resampled_train_dataset, dataset_title="Training dataset", bins=bins)
    plot_value_histogram(resampled_test_dataset, dataset_title="Testing dataset", bins=bins)
    if resample:
        plt.savefig(os.path.join(result_path, "image_values_" +
                    str(bins) + "_bins_resampled.png"))
    else:
        plt.savefig(os.path.join(
            result_path, "image_values_" + str(bins) + "_bins.png"))

    # Label histograms
    n = len(label_names)
    if resample:
        train_labels = [label for (_, label) in resampled_train_dataset]
        test_labels = [label for (_, label) in resampled_test_dataset]

    else:
        train_labels = [label for (_, label) in train_dataset]
        test_labels = [label for (_, label) in test_dataset]

    plt.figure()
    plt.title("Example distribution by class")
    fig, axes = plt.subplots(nrows=2, ncols=1)
    plt.xlabel("Classes")
    axes[0].set_ylabel("Training samples")
    axes[0].hist(train_labels, rwidth=0.7, bins=np.arange(-0.5, n + 0.5, 1.0))

    # Plot examples
    axes[1].set_ylabel("Testing samples")
    axes[1].hist(test_labels, rwidth=0.7, bins=np.arange(-0.5, n + 0.5, 1.0))
    if resample:
        plt.savefig(os.path.join(
            result_path, "example_distribution_by_class_resampled.png"))
    else:
        plt.savefig(os.path.join(
            result_path, "example_distribution_by_class.png"))

    if plot_images:
        plot_examples_per_label(train_dataset, label_names,
                                object_size=params.object_size, num_examples=5,
                                image_label_indices=train_data[1], title="train examples", result_path=result_path)
        plot_examples_per_label(test_dataset, label_names, object_size=params.object_size,
                                num_examples=5, image_label_indices=test_data[1], title="test examples", result_path=result_path)
    else:
        plot_examples_per_label(train_dataset, label_names, object_size=params.object_size,
                                num_examples=5, title="train examples", result_path=result_path)
        plot_examples_per_label(test_dataset, label_names, object_size=params.object_size,
                                num_examples=5, title="test examples", result_path=result_path)


def plot_examples_per_label(dataset, label_names, object_size=(300, 500), num_examples=10, image_label_indices=None, title="examples", result_path=""):
    n = len(label_names)
    labels = list(range(n))

    if image_label_indices is not None:
        step = 2
    else:
        step = 1
    num_windows = num_examples * step
    fig, axes = plt.subplots(nrows=n, ncols=num_windows, figsize=(12, 8))
    fig.suptitle(title)
    indices_by_label = group_indices_by_label(dataset, labels)

    for ax, row in zip(axes[:, 0], label_names):
        ax.set_ylabel(row, rotation=0, size='large')
    for label_idx in range(n):
        for i in range(0, min(num_examples, len(indices_by_label[label_idx]))):
            example_index = indices_by_label[label_idx][i]
            fig = axes[label_idx, i *
                       step].imshow(dataset[example_index][0][0])
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            if image_label_indices is not None:
                object = kitti_utils.extract_object_from_image(
                    "%06d" % image_label_indices[example_index][0], image_label_indices[example_index][1], object_size=object_size)
                fig = axes[label_idx, i * step + 1].imshow(object)
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
        axes[label_idx, 0].axes.get_xaxis().set_visible(True)
        axes[label_idx, 0].axes.get_yaxis().set_visible(True)
        if image_label_indices is not None:
            axes[label_idx, 1].axes.get_xaxis().set_visible(True)
            axes[label_idx, 1].axes.get_yaxis().set_visible(True)

    plt.savefig(os.path.join(result_path, title + ".png"))


def load_dataset(batch_size, dataset_path="../data/processed/", show=False, resample=False):
    print(dataset_path + "training.pt")
    train_dataset = torch.load(dataset_path + "training.pt")[0]
    test_dataset = torch.load(dataset_path + "testing.pt")[0]


    if resample:
        train_sampler = get_dataset_sampler(train_dataset)
        test_sampler = get_dataset_sampler(test_dataset)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  sampler=test_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        # Do not shuffle the test dataset as it should match the metadata
        # used for analysis
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
    if show:
        (imgs, _) = next(iter(train_loader))
        grid_img = torchvision.utils.make_grid(
            imgs[:10], nrow=5, pad_value=255)
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()

    return train_loader, test_loader

if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="../data/processed/",
                        help='Directory of the processed KITTI dataset.')

    FLAGS, unparsed = parser.parse_known_args()
    analyse_kitti_objects_dataset(FLAGS.dataset_path, resample=False, plot_images=True)
    analyse_kitti_objects_dataset(FLAGS.dataset_path, resample=True, plot_images=True)
