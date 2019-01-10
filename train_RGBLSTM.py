import os
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
from AdamW import AdamW
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

input_dim = 2048
batchsize = 128
hidden_dim = 512
num_class = 101
learning_rate = 1e-3
epoch = 90
memory_size = 2000
# torch.backends.cudnn.enabled=False
pretrain = True
frame_length = 200

CEL_loss = nn.CrossEntropyLoss()
model = LSTM_classifier(input_dim, hidden_dim, num_class, batchsize)
model = model.cuda()

memory = LSTM_with_memory(input_dim, hidden_dim, memory_size, num_class, batchsize)
memory = memory.cuda()

root_dir = '/media/kongyu/Data10TB/UCF-101-kev1n/LSTM_Memory/RGB/train/'
test_dir = '/media/kongyu/Data10TB/UCF-101-kev1n/LSTM_Memory/RGB/val/'
path_for_model = '/media/kongyu/SSD1TB/kev1n/LSTM_memory_6/model/'

end_with = 'npy'
train_data = UCF101_Dataset(root_dir, frame_length,end_with)
test_data = UCF101_Dataset(test_dir, frame_length,end_with)
dataloader = data.DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=2)
test_loader = data.DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=2)
num_batch = len(dataloader)

params = [
    {'params': model.parameters()},
]

best_result = 0

optimizer = optim.Adam(params, lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
if pretrain:
    for e in range(epoch):
        running_loss = 0.0
        for x_batch, y_batch in dataloader:
            current_batchsize = x_batch.size(0)
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

            model.clear_history(current_batchsize)

            # x_batch = x_batch.transpose(0,1)
            prediction, penalty = model(x_batch)
            loss = CEL_loss(prediction, y_batch) + 0.001 * penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.data[0]

            running_loss += batch_loss
        model.eval()
        train_accuracy = accuracy(dataloader, model)
        test_accuracy = accuracy(test_loader, model)
        model.train()

        if (test_accuracy >= best_result):
            best_result = test_accuracy
            torch.save(model.state_dict(), os.path.join(path_for_model, 'rgb_lstm.pt'))
        print('Epoch [%d], Loss: %.4f, Train accuracy: %.4f, Test accuracy: %.4f, Best: %.4f' % (
        e, running_loss / num_batch, train_accuracy, test_accuracy, best_result))

state_dict = torch.load(os.path.join(path_for_model, 'rgb_lstm.pt'))
model.load_state_dict(state_dict)

# memory.lstm1.weight_il_l0.data.copy_(model.lstm1.weight_il_l0.data)
# memory.lstm2.weight_il_l0.data.copy_(model.lstm2.weight_il_l0.data)


for w1, w2 in zip(memory.lstm1.parameters(), model.lstm1.parameters()):
  w1.data.copy_(w2.data)

memory.batchnorm.weight.data = model.batchnorm.weight.data

retrain_rate = 1e-3

mem_params = [
    {'params': memory.memory.parameters()},
    {'params': memory.lstm1.parameters(), 'lr': retrain_rate*0.1},
    {'params': memory.batchnorm.parameters(), 'lr': retrain_rate*0.1}
 ]

# optimizer = AdamW(params, lr=learning_rate, weight_decay=0.001)
optimizer = optim.Adam(params, lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for e in range(epoch):
    running_loss = 0.0
    for x_batch, y_batch in dataloader:
        current_batchsize = x_batch.size(0)

        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        memory.clear_history(current_batchsize)

        x_batch = x_batch.transpose(0, 1)

        y_hat, softmax_score, loss = memory(x_batch, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.data[0]

        running_loss += batch_loss
    memory.eval()
    train_accuracy = accuracy_memory(dataloader, memory)
    test_accuracy = accuracy_memory(test_loader, memory)
    memory.train()
    if (test_accuracy >= best_result):
        best_result = test_accuracy
        torch.save(memory, os.path.join(path_for_model, 'rgb_memory.pt'))
    print('Epoch [%d], Loss: %.4f, Train accuracy: %.4f, Test accuracy: %.4f, Best: %.4f' % (
    e, running_loss / num_batch, train_accuracy, test_accuracy, best_result))
