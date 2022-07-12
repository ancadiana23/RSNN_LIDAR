import argparse
import matplotlib.pyplot as plt
import pprint
import torch

from datetime import datetime
from torch.optim.lr_scheduler import StepLR

from dataset import load_dataset
from SNN import SNN
import train


def save_image_with_colorbar(image, path):
    plt.figure()
    im = plt.imshow(image)
    plt.colorbar(im)
    plt.savefig(path)


def main(params):
    if params.print:
        print(params)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_loader, test_loader = load_dataset(params.batch_size,
                                             dataset_path=params.data_path,
                                             resample=False)
    resampled_train_loader, resampled_test_loader = load_dataset(
        params.batch_size, dataset_path=params.data_path, resample=True)

    output_dim = 7
    input_dim = params.kernel_size[0] * params.kernel_size[1]
    if (params.encoding == "poisson" or "RBF" in params.encoding) and params.neurons_per_pixel != 0:
        input_dim *= params.neurons_per_pixel

    model = SNN(input_dim, params.hidden_dims, output_dim, device,
                       params.b_j0, dt=params.dt, R_m=params.R_m,
                       sparsity=params.sparsity)
    print(model)
    model.to(device)

    optimizer = torch.optim.Adam([{'params': model.get_base_parameters()}],
                                lr=params.init_lr)
    scheduler = StepLR(optimizer, step_size=params.lr_step_size,
                       gamma=params.lr_decay)
    print('Time: ', datetime.now().strftime("%d-%m-%Y %H:%M:%S"))

    # Train on the resampled training set, while testing on the original testing dataset.
    if params.training == "BPTT":
        train_acc, test_acc = train.train(model, params.num_epochs,
                                          resampled_train_loader, test_loader, optimizer, scheduler, device,
                                          kernel_size=params.kernel_size, stride=params.stride,
                                          encoding=params.encoding, time_per_window=params.time_per_window,
                                          neurons_per_pixel=params.neurons_per_pixel)

    elif params.training == "TBPTT":
        train_acc, test_acc = train.train_TBPTT(model, params.num_epochs,
                                                   resampled_train_loader, test_loader, optimizer, scheduler, device,
                                                   kernel_size=params.kernel_size, stride=params.stride,
                                                   encoding=params.encoding,
                                                   time_per_window=params.time_per_window,
                                                   neurons_per_pixel=params.neurons_per_pixel,
                                                   backprop_step=params.backprop_step)

    # Construct confusion matrix using the resampled testing dataset for a
    # more clear visualization.
    test_accuracy, test_confusion_matrix = train.test(model, test_loader, device,
                                                      kernel_size=params.kernel_size, stride=params.stride,
                                                      encoding=params.encoding,
                                                      time_per_window=params.time_per_window,
                                                      neurons_per_pixel=params.neurons_per_pixel)
    resampled_test_accuracy, resampled_test_confusion_matrix = train.test(model, resampled_test_loader, device,
                                                                          kernel_size=params.kernel_size,
                                                                          stride=params.stride,
                                                                          encoding=params.encoding,
                                                                          time_per_window=params.time_per_window,
                                                                          neurons_per_pixel=params.neurons_per_pixel)

    print('Test Accuracy: ', test_accuracy)
    print('Resampled Test Accuracy: ', resampled_test_accuracy)
    print('Time: ', datetime.now().strftime("%d-%m-%Y %H:%M:%S"))

    torch.save((model, params), params.model_path)

    save_image_with_colorbar(test_confusion_matrix,
                             params.confusion_matrix_path + ".png")
    save_image_with_colorbar(resampled_test_confusion_matrix,
                             params.confusion_matrix_path + "_resampled.png")

    ###################
    # Accuracy  curve
    ###################
    plt.figure()
    plt.plot(train_acc, label="Train accuracy")
    plt.plot(test_acc, label="Test accuracy")
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim((0.0, 100.0))
    plt.legend()
    plt.savefig(params.plot_name)

    return test_accuracy


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--b_j0', type=float, default=0.01,
                        help='Neural threshold baseline')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Batch size to run trainer.')
    parser.add_argument('--backprop_step', type=int, default=10,
                        help='Number of steps to truncate the BPTT to.')
    parser.add_argument('--confusion_matrix_path', type=str, default="../results/confusion_matrix",
                        help='Path to save the confusion_matrix.')
    parser.add_argument('--data_path', type=str, default="../data/processed/",
                        help='Path to dataset directory.')
    parser.add_argument('--dt', type=float, default=10.0, help='Time period.')
    parser.add_argument('--encoding', type=str, default="rate",
                        help='Input encoding method; options: {"rate", "temporal", "poisson", "RBF", "RBF_TC"}.')
    parser.add_argument('--hidden_dims', type=int, default=[128, 64], nargs="+",
                        help='Dimention of hidden layers.')
    parser.add_argument('--init_lr', type=float, default=0.01,
                        help='Initial value for the learning rate.')
    parser.add_argument('--kernel_size', type=int, default=[1, 50], nargs="+",
                        help='Size of the sliding window over the image. Used to turn image into sequence.')
    parser.add_argument('--lr_decay', type=float, default=0.9,
                        help='Learning rate decay.')
    parser.add_argument('--lr_step_size', type=int, default=10,
                        help='Number of epochs after which the learning rate will be changed.')
    parser.add_argument('--model_path', type=str, default="../results/model.pt",
                        help='Path for saving the trained model.')
    parser.add_argument('--neurons_per_pixel', type=int, default=0,
                        help='Number of input neurons per pixel for the poisson encoding.')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--plot_name', type=str, default="../results/training_accuracy.png",
                        help="Path for saving the training accuracy plot.")
    parser.add_argument('--R_m', type=float, default=1.0,
                        help='Neural membrane resistance')
    parser.add_argument('--stride', type=int, default=[1, 50], nargs="+",
                        help='Stride of the sliding window over the image. Used to turn image into sequence.')
    parser.add_argument('--training', type=str, default="BPTT",
                        help='Training type. Options: {"BPTT", "TBPTT"}')
    parser.add_argument('--sparsity', type=float, default=0.0,
                        help='Desired parameter sparsity.')
    parser.add_argument('--print', action='store_true', help='Verbose.')
    parser.add_argument('--time_per_window', type=int,
                        default=0, help='Desired parameter sparsity.')

    FLAGS, unparsed = parser.parse_known_args()
    accuracy = main(FLAGS)
