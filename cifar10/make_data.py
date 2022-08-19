import torchvision
import os
import json
from PIL import Image



download_path = 'datasets'
if not os.path.exists(download_path):
    os.makedirs(download_path)
trainset = torchvision.datasets.CIFAR10(root=download_path, train=True, download=True)
testset = torchvision.datasets.CIFAR10(root=download_path, train=False, download=True)


for dataset in [trainset, testset]:
    data_path = 'datasets/datasets/cifar10/cifar10/' + ('train/' if dataset.train else 'test/')
    for (idx,sample) in enumerate(dataset):
        img, label = sample
        clase_name = dataset.classes[label]
        folder_path = data_path + clase_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_path = os.path.join(folder_path, '{}_{}.png'.format(idx, clase_name))
        image = img.convert('RGB')
        image.save(file_path)
