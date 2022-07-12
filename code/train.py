import gc
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cutorch

from kitti_utils import get_all_labels
import utils


def train(model, num_epochs, train_loader, test_loader, optimizer, scheduler,
          device, kernel_size=(1, 4), stride=(1, 1), encoding="rate", time_per_window=0, neurons_per_pixel=0):
    train_accuracies = []
    test_accuracies = []
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            windows = utils.encode_data(images, kernel_size, stride, device,
                                        encoding=encoding,
                                        time_per_window=time_per_window,
                                        neurons_per_pixel=neurons_per_pixel)
            labels = labels.long().to(device)
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output/logits
            outputs = model(windows)
            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()
            model.reset_weight_sparsity()
            del windows
            gc.collect()

        scheduler.step()
        train_accuracy, _ = test(model, train_loader, device,
                                 kernel_size=kernel_size, stride=stride, encoding=encoding, time_per_window=time_per_window,
                                 neurons_per_pixel=neurons_per_pixel)
        test_accuracy, _ = test(model, test_loader, device,
                                kernel_size=kernel_size, stride=stride, encoding=encoding, time_per_window=time_per_window,
                                neurons_per_pixel=neurons_per_pixel)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        res_str = 'epoch: ' + str(epoch) + ' Loss: ' + str(loss.item()) + '. Tr Accuracy: ' + \
            str(train_accuracy) + '. Ts Accuracy: ' + str(test_accuracy)
        print(res_str)
    return train_accuracies, test_accuracies


def train_TBPTT(model, num_epochs, train_loader, test_loader, optimizer,
                   scheduler, device, kernel_size=(1, 4), stride=(1, 1), encoding="rate",
                   time_per_window=0, neurons_per_pixel=0, backprop_step=16):
    train_accuracies = []
    test_accuracies = []
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            windows = utils.encode_data(images, kernel_size, stride, device,
                                        encoding=encoding,
                                        time_per_window=time_per_window,
                                        neurons_per_pixel=neurons_per_pixel)

            labels = labels.long().to(device)
            optimizer.zero_grad()  # Clear gradients w.r.t. parameters
            _, loss = model.forward_TBPTT(windows, backprop_step=backprop_step,
                                          train=True, labels=labels,
                                          criterion=criterion, optimizer=optimizer)
        scheduler.step()
        train_accuracy, _ = test(model, train_loader, device,
                                 kernel_size=kernel_size, stride=stride,
                                 encoding=encoding, time_per_window=time_per_window,
                                 neurons_per_pixel=neurons_per_pixel)
        test_accuracy, _ = test(model, test_loader, device,
                                kernel_size=kernel_size, stride=stride,
                                encoding=encoding, time_per_window=time_per_window,
                                neurons_per_pixel=neurons_per_pixel)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        res_str = 'epoch: ' + str(epoch) + ' Loss: ' + str(loss.item()) + \
            '. Tr Accuracy: '+str(train_accuracy) + \
            '. Ts Accuracy: ' + str(test_accuracy)
        print(res_str)
    return train_accuracies, test_accuracies


def test(model, dataloader, device, kernel_size=(1, 4), stride=(1, 1),
         encoding="rate", time_per_window=0, neurons_per_pixel=0, model_name="",
         directory="", postfix=""):
    correct = 0
    total = 0
    num_classes = model.output_dim
    confusion_matrix = torch.zeros((num_classes, num_classes))

    for images, labels in dataloader:
        windows = utils.encode_data(images, kernel_size, stride, device,
                                    encoding=encoding,
                                    time_per_window=time_per_window,
                                    neurons_per_pixel=neurons_per_pixel)

        outputs = model(windows)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        if torch.cuda.is_available():
            correct += (predicted.cpu() == labels.long().cpu()).sum()
        else:
            correct += (predicted == labels).sum()
        for i, label in enumerate(labels):
            confusion_matrix[label][predicted[i]] += 1

    accuracy = 100. * correct.numpy() / total
    return accuracy, confusion_matrix
