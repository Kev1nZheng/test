import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from LSTM_structure import LSTM_classifier, domain_fusion_memory
from UCF_101 import UCF101_fusion
from utils import accuracy, crossdomain_accuracy
from AdamW import AdamW
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

input_dim = 2048
batchsize = 128
hidden_dim = 512
num_class = 101
learning_rate = 1e-4
epoch = 90
memory_size = 2000
frame_length = 200

CEL_loss = nn.CrossEntropyLoss()
rgb_lstm = LSTM_classifier(input_dim, hidden_dim, num_class, batchsize)
flow_lstm = LSTM_classifier(input_dim, hidden_dim, num_class, batchsize)
rgb_lstm = rgb_lstm.cuda()
flow_lstm = flow_lstm.cuda()
memory = domain_fusion_memory(input_dim, hidden_dim, memory_size, num_class, batchsize)
memory = memory.cuda()

root_dir_flow = '/media/kongyu/Black6TB/Data/UCF-101/FlowLSTM/train/'
root_dir_rgb = '/media/kongyu/Black6TB/Data/UCF-101/RGBLSTM/train/'
test_dir_flow = '/media/kongyu/Black6TB/Data/UCF-101/FlowLSTM/val/'
test_dir_rgb = '/media/kongyu/Black6TB/Data/UCF-101/RGBLSTM/val/'
path_for_model = '/media/kongyu/Black6TB/kev1n/LSTM_memory_5/model/'
end_with = 'npy'

state_dict = torch.load(os.path.join(path_for_model, 'rgb_200_lstm.pt'))
rgb_lstm.load_state_dict(state_dict)
state_dict = torch.load(os.path.join(path_for_model, 'flow_200_lstm.pt'))
flow_lstm.load_state_dict(state_dict)

for w1, w2 in zip(memory.lstm_rgb.parameters(), rgb_lstm.lstm1.parameters()):
    w1.data.copy_(w2.data)

for w1, w2 in zip(memory.lstm_flow.parameters(), flow_lstm.lstm1.parameters()):
    w1.data.copy_(w2.data)

memory.batchnorm_rgb.weight.data = rgb_lstm.batchnorm.weight.data
memory.batchnorm_flow.weight.data = flow_lstm.batchnorm.weight.data

train_data = UCF101_fusion(root_dir_rgb, root_dir_flow, frame_length, end_with)
test_data = UCF101_fusion(test_dir_rgb, test_dir_flow, frame_length, end_with)
dataloader = data.DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=2)
test_loader = data.DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=2)
num_batch = len(dataloader)

params = [
    {'params': memory.memory.parameters()},
    {'params': memory.batchnorm_rgb.parameters(), 'lr': learning_rate * 0.1},
    {'params': memory.batchnorm_flow.parameters(), 'lr': learning_rate * 0.1},
    {'params': memory.lstm_rgb.parameters(), 'lr': learning_rate * 0.1},
    {'params': memory.lstm_flow.parameters(), 'lr': learning_rate * 0.1},
]

best_result = 0

# optimizer = AdamW(params, lr=learning_rate, weight_decay=0.005)
optimizer = optim.Adam(params, lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for e in range(epoch):
    running_loss = 0.0
    for rgb_batch, flow_batch, y_batch in dataloader:
        current_batchsize = rgb_batch.size(0)

        rgb_batch, flow_batch, y_batch = rgb_batch.cuda(), flow_batch.cuda(), y_batch.cuda()

        memory.clear_history(current_batchsize)

        y_hat, loss, s_penalty = memory(rgb_batch, flow_batch, y_batch)
        # loss = loss + 0.001*s_penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.data[0]

        running_loss += batch_loss
    memory.eval()
    train_accuracy = crossdomain_accuracy(dataloader, memory)
    test_accuracy = crossdomain_accuracy(test_loader, memory)
    memory.train()
    if (test_accuracy >= best_result):
        best_result = test_accuracy
        # torch.save(memory.state_dict(), os.path.join(path_for_model, 'memory1_lstm.pt'))
        torch.save(memory,os.path.join(path_for_model, 'memory_200_lstm.pt'))
    print('Epoch [%d], Loss: %.4f, Train accuracy: %.4f, Test accuracy: %.4f, Best: %.4f' % (
        e, running_loss / num_batch, train_accuracy, test_accuracy, best_result))
# memory_test = torch.load(os.path.join(path_for_model, 'memory2_lstm.pt'))
# # memory_state_dict = torch.load(os.path.join(path_for_model, 'memory1_lstm.pt'))
# # memory_test.load_state_dict(memory_state_dict)
# memory_test = memory_test.cuda()
# memory_test.eval()
# test_accuracy = crossdomain_accuracy(test_loader, memory_test)
# print('Test accuracy: %.4f' % (test_accuracy))