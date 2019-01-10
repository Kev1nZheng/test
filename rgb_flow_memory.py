import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from LSTM_structure import LSTM_classifier, domain_fusion_memory
from UCF_101 import UCF101_fusion_Clips
from utils import accuracy, crossdomain_accuracy
from AdamW import AdamW
import os
from UCF_101 import UCF101_Dataset_Clips
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

input_dim = 2048
batchsize = 1
hidden_dim = 512
num_class = 101
learning_rate = 1e-3
epoch = 30
memory_size = 2000


def calaculare_score(net, test_loader):
    total = 0.0
    correct = 0.0
    out_score = np.empty((len(rgb_test_loader), 101))
    with torch.no_grad():
        for input, labels in test_loader:
            input = input.cuda()
            current_batchsize = input.size(0)
            net.clear_history(current_batchsize)
            score = net(input)
            if type(score) == tuple:
                score = score[0]
            score = F.softmax(score, dim=1)
            _, preds = score.max(1)
            score = score.cpu()
            score = score.data.numpy()
            out_score[int(total), :] = score[0, :]
            total += labels.size(0)
            correct += (preds.cpu().data == labels).sum().item()
    return out_score, correct / total


rgb_test_dir = '/media/kongyu/Black6TB/Data/UCF-101/RGBLSTM/val/'
flow_test_dir = '/media/kongyu/Black6TB/Data/UCF-101/FlowLSTM/val/'
path_for_model = '/media/kongyu/Black6TB/kev1n/LSTM_memory_5/model/'
end_with = 'npy'

rgb_model = LSTM_classifier(input_dim, hidden_dim, num_class, batchsize)
rgb_state_dict = torch.load(os.path.join(path_for_model, 'rgb_200_lstm.pt'))
rgb_model.load_state_dict(rgb_state_dict)
params = [
    {'params': rgb_model.parameters()},
]
rgb_model = rgb_model.cuda()
rgb_model.eval()

flow_model = LSTM_classifier(input_dim, hidden_dim, num_class, batchsize)
flow_state_dict = torch.load(os.path.join(path_for_model, 'flow_200_lstm.pt'))
flow_model.load_state_dict(flow_state_dict)
params = [
    {'params': flow_model.parameters()},
]
flow_model = flow_model.cuda()
flow_model.eval()

memory = torch.load(os.path.join(path_for_model, 'memory_200_lstm.pt'))
memory = memory.cuda()
memory.eval()

for clip in range(1, 11):
    rgb_test_data = UCF101_Dataset_Clips(rgb_test_dir, clip, end_with)
    rgb_test_loader = data.DataLoader(rgb_test_data, batch_size=batchsize, shuffle=False, num_workers=2)

    flow_test_data = UCF101_Dataset_Clips(flow_test_dir, clip, end_with)
    flow_test_loader = data.DataLoader(flow_test_data, batch_size=batchsize, shuffle=False, num_workers=2)

    memory_test_data = UCF101_fusion_Clips(rgb_test_dir, flow_test_dir, clip, end_with)
    memory_test_loader = data.DataLoader(memory_test_data, batch_size=batchsize, shuffle=False, num_workers=2)

    total = 0.0
    correct = 0.0
    memory_score = np.empty((len(memory_test_loader), 101))
    with torch.no_grad():
        for images_1, images_2, labels in memory_test_loader:
            images_1 = images_1.cuda()
            images_2 = images_2.cuda()
            current_batchsize = images_1.size(0)
            memory.clear_history(current_batchsize)
            y_hat, softmax_score = memory.predict(images_1, images_2)
            memory_score[int(total), :] = softmax_score[0, :]
            total += labels.size(0)
            correct += (y_hat.squeeze().cpu().data == labels).sum().item()
    memory_accuracy = correct / total

    rgb_score, rgb_accuracy = calaculare_score(rgb_model, rgb_test_loader)
    flow_score, flow_accuracy = calaculare_score(flow_model, flow_test_loader)
    # print(rgb_score.std())
    # print(flow_score.std())
    print(memory_score)
    total = 0.0
    correct = 0.0
    fusion_3_score = (rgb_score + flow_score) + memory_score
    f3score = np.empty((1, 101))
    fusion_3_score = torch.from_numpy(fusion_3_score)
    f3score = torch.from_numpy(f3score)
    with torch.no_grad():
        for input, labels in flow_test_loader:
            f3score[0, :] = fusion_3_score[int(total), :]
            _, preds = f3score.max(1)
            total += labels.size(0)
            correct += (preds.cpu().data == labels).sum().item()
    fusion_3_accuracy = correct / total

    total = 0.0
    correct = 0.0
    fusion_2_score = rgb_score + flow_score
    f2score = np.empty((1, 101))
    fusion_2_score = torch.from_numpy(fusion_2_score)
    f2score = torch.from_numpy(f2score)
    with torch.no_grad():
        for input, labels in flow_test_loader:
            f2score[0, :] = fusion_2_score[int(total), :]
            _, preds = f2score.max(1)
            total += labels.size(0)
            correct += (preds.cpu().data == labels).sum().item()
    fusion_2_accuracy = correct / total
    print('Clip:{0} RGB:{1:.4} Flow:{2:.4} Memory:{3:.5} Classifier:{4:.5} Full:{5:.5}'.format(clip, rgb_accuracy,
                                                                                               flow_accuracy,
                                                                                               memory_accuracy,
                                                                                               fusion_2_accuracy,
                                                                                               fusion_3_accuracy))
