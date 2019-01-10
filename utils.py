import torch
import torch.nn.functional as F
import numpy as np


def score_predict(model, x):
    score = model(x)
    if type(score) == tuple:
        score = score[0]
    score = F.softmax(score, dim=1)
    _, prediction = score.max(1)

    return prediction


def accuracy(iter, model):
    total = 0.0
    correct = 0.0

    with torch.no_grad():
        for input, labels in iter:
            input = input.cuda()
            current_batchsize = input.size(0)
            # input = input.transpose(0,1)
            model.clear_history(current_batchsize)
            preds = score_predict(model, input)
            total += labels.size(0)
            correct += (preds.cpu().data == labels).sum().item()

    return correct / total


def accuracy_memory(datloader, m):
    total = 0.0
    correct = 0.0

    with torch.no_grad():
        for input, labels in datloader:
            input = input.cuda()
            current_batchsize = input.size(0)
            input = input.transpose(0, 1)
            m.clear_history(current_batchsize)

            y_hat, softmax_score = m.predict(input)

            total += labels.size(0)
            correct += (y_hat.squeeze().cpu().data == labels).sum().item()

    return correct / total


def convert_softmax(num_class, top_softmax):
    return


def crossdomain_score(model_1, model_2, x_1, x_2):
    score_1, _ = model_1(x_1)
    score_2, _ = model_2(x_2)
    score = score_1 + score_2
    score = F.softmax(score / 2, dim=1)
    _, prediction = score.max(1)

    return prediction


def crossdomain_accuracy(iter, m):
    total = 0.0
    correct = 0.0

    with torch.no_grad():
        for images_1, images_2, labels in iter:
            images_1 = images_1.cuda()
            images_2 = images_2.cuda()
            current_batchsize = images_1.size(0)
            m.clear_history(current_batchsize)
            y_hat, softmax_score = m.predict(images_1, images_2)

            # _, prediction = softmax_score.max(1)
            total += labels.size(0)
            correct += (y_hat.squeeze().cpu().data == labels).sum().item()
            # correct += (prediction.cpu().data == labels).sum().item()
            # print(total,correct)
    return correct / total


def unique(tensor1d):
    t, _ = np.unique(tensor1d.cpu().numpy(), return_inverse=True)
    return torch.from_numpy(t)
