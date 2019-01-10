import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from LSTM_structure import LSTM_classifier, LSTM_with_memory
from UCF_101 import UCF101_Dataset
from utils import accuracy, accuracy_memory
import timeit
import os
import numpy as np

input_dim = 2048
batchsize = 1
hidden_dim = 512
num_class = 101
learning_rate = 1e-3
epoch = 30
memory_size = 2000
# torch.backends.cudnn.enabled=False
pretrain = True

rgb_test_dir = '/media/kongyu/Black6TB/Data/UCF-101/RGBLSTM/val/'
flow_test_dir = '/media/kongyu/Black6TB/Data/UCF-101/FlowLSTM/val/'

path_for_model = '/media/kongyu/Black6TB/kev1n/LSTM_memory_F/model/'
end_with = 'npy'

rgb_test_data = UCF101_Dataset(rgb_test_dir, end_with)
rgb_test_loader = data.DataLoader(rgb_test_data, batch_size=batchsize, shuffle=False, num_workers=2)
flow_test_data = UCF101_Dataset(flow_test_dir, end_with)
flow_test_loader = data.DataLoader(flow_test_data, batch_size=batchsize, shuffle=False, num_workers=2)

rgb_model = LSTM_classifier(input_dim, hidden_dim, num_class, batchsize)
rgb_state_dict = torch.load(os.path.join(path_for_model, 'rgb_lstm.pt'))
rgb_model.load_state_dict(rgb_state_dict)
params = [
    {'params': rgb_model.parameters()},
]
rgb_model = rgb_model.cuda()
rgb_model.eval()

flow_model = LSTM_classifier(input_dim, hidden_dim, num_class, batchsize)
flow_state_dict = torch.load(os.path.join(path_for_model, 'flow_lstm.pt'))
flow_model.load_state_dict(flow_state_dict)
params = [
    {'params': flow_model.parameters()},
]
flow_model = flow_model.cuda()
flow_model.eval()

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
    print('Accuracy:{}'.format(correct / total))
    return out_score

rgb_score = calaculare_score(rgb_model,rgb_test_loader)
flow_score = calaculare_score(flow_model,flow_test_loader)


total = 0.0
correct = 0.0
fusion_score = rgb_score * 0.55 + flow_score * 0.45
fscore = np.empty((1, 101))
fusion_score = torch.from_numpy(fusion_score)
fscore = torch.from_numpy(fscore)
with torch.no_grad():
    for input, labels in flow_test_loader:
        fscore[0, :] = fusion_score[int(total), :]
        _, preds = fscore.max(1)
        total += labels.size(0)
        correct += (preds.cpu().data == labels).sum().item()
print('Fusion Accuracy:', correct / total)
