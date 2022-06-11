import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils import split_data
from model import VGG16_pretrained_model
from orchid import Orchid
from trainer import train
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

if __name__ == '__main__':

    # Parser initializing
    parser = argparse.ArgumentParser(description="Orchid classification")
    parser.add_argument('--ngpu', default=0, type=int, required=False)
    parser.add_argument('--label_path', default="dataset", type=str, required=False)
    parser.add_argument('--img_path', default="dataset/data", type=str, required=False)
    parser.add_argument('--batch_size', default=32, type=int, required=False)
    args = parser.parse_args()

    numClasses = 219

    # Device
    device = torch.device("cuda:0")


    # 分dataset 成 8 train 2 valid
    print("splitting dataset")
    split_data()

    # 分開讀取read dataset
    print("loading dataset")
    inputSize = 224
    dataTransformsTrain = transforms.Compose([
        transforms.Resize((inputSize, inputSize)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    #
    # dataAugTransformTrain = transforms.Compose([
    #     transforms.RandomCrop(size=(224, 224)),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomverticalFlip(p=0.5),
    #     transforms.RandomRotation(30),
    #     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    dataTransformsValid = transforms.Compose([
        transforms.Resize((inputSize, inputSize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trainSet = Orchid(args.label_path + "/train.csv", args.img_path, dataTransformsTrain)
    # for i in range(4):
    #     temDataset = Orchid(args.label_path + "/train.csv", args.img_path, dataAugTransformTrain)
    #     trainSet = ConcatDataset([temDataset, trainSet])

    validSet = Orchid(args.label_path + "/valid.csv", args.img_path, dataTransformsValid)

    # dataLoader
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validLoader = torch.utils.data.DataLoader(validSet, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("loading dataset complete")

    # load Model
    model = VGG16_pretrained_model(numClasses, featureExtract=True, usePretrained=True).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    criterion = nn.CrossEntropyLoss().to(device)

    model_ft = train(model, trainLoader, validLoader, optimizer, criterion, 10, device)
