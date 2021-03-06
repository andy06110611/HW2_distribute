import numpy as np
import torch
from utils import saveModel
import tqdm


def train(net, trainLoader, testLoader, optimizer, criterion, epochs, device):
    net.train()
    testAccuracy = 0
    bestModel = net
    for i in range(epochs):
        totalLoss = 0
        accuracy = 0
        count = 0
        for img, label in trainLoader:
            img = img.to(device)
            label = label.to(device, dtype=torch.long)

            # zero the parameter gradients
            optimizer.zero_grad()

            output = net(img)
            loss = criterion(output, label)
            _, predicted = torch.max(output.data, 1)
            count += len(img)
            accuracy += (predicted == label).sum().item()
            totalLoss += loss.item()*len(label)
            loss.backward()
            optimizer.step()

        print("------------------------------------------------------")
        print("train lost")
        print("epoch:" + str(i) + "/" + str(epochs))
        print("Train Loss: {}".format(totalLoss / count))
        print("Train Accuracy: {}".format(accuracy / count))
        print("------------------------------------------------------")

        if (i % 10 == 0):
            tmpAccuracy = test(net, testLoader, criterion, device)
            if (tmpAccuracy > testAccuracy):
                testAccuracy = tmpAccuracy
                bestModel = net
                epoch = i


    saveModel(bestModel, "model/epoch" + str(epoch) + "_" + str(testAccuracy)+".pth")
    return net


def test(net, testLoader, criterion, device):
    net.eval()
    totalLoss = 0
    accuracy = 0
    count = 0
    for x, label in testLoader:
        x = x.to(device)
        label = label.to(device, dtype=torch.long)
        output = net(x)
        loss = criterion(output, label)
        _, predicted = torch.max(output.data, 1)
        count += len(x)
        accuracy += (predicted == label).sum().item()
        totalLoss += loss.item() * len(label)
    print("------------------------------------------------------")
    print("Test Loss: {}".format(totalLoss / count))
    print("Test Accuracy: {}".format(accuracy / count))
    print("------------------------------------------------------")
    return (accuracy / count)