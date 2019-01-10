import torch
import torch.nn as nn
import torch.nn.functional as F
from memory import Memory


class LSTM_classifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_class, batchsize):
        super(LSTM_classifier, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm1 = nn.LSTM(input_dim, hidden_dim)
        # The linear layer that maps from hidden state space to tag space
        self.clssifier = nn.Linear(hidden_dim, num_class)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.batchsize = batchsize

        self.hidden1 = self.init_hidden()

    def init_hidden(self, single_batchsize=None):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if single_batchsize == None:
            return (torch.zeros(1, self.batchsize, self.hidden_dim).cuda(),
                    torch.zeros(1, self.batchsize, self.hidden_dim).cuda())
        else:
            return (torch.zeros(1, single_batchsize, self.hidden_dim).cuda(),
                    torch.zeros(1, single_batchsize, self.hidden_dim).cuda())

    def clear_history(self, current_batchsize):

        self.hidden1 = self.init_hidden(current_batchsize)

    def forward(self, sequence):
        time_step = sequence.size(1)
        batchsize = sequence.size(0)

        sequence = sequence.transpose(0, 1)
        sequence = sequence.float()
        # lstm1_out, self.hidden1 = self.lstm1(sequence, self.hidden1)
        lstm1_out, self.hidden1 = self.lstm1(sequence, self.hidden1)

        # sequence = sequence.mean(0)
        stablize_penalty = stablizing_rnn(lstm1_out)

        average_out = lstm1_out.mean(0)
        # average_out = sequence
        # average_out = self.linear(average_out)
        average_out = self.batchnorm(average_out)
        average_out = F.relu(average_out)
        prediction = self.clssifier(average_out)

        return prediction, stablize_penalty


class domain_fusion_memory(nn.Module):
    def __init__(self, input_dim, hidden_dim, memory_size, num_class, batchsize):
        super(domain_fusion_memory, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm_rgb = nn.LSTM(input_dim, hidden_dim)
        self.lstm_flow = nn.LSTM(input_dim, hidden_dim)
        self.memory = Memory(memory_size, 2 * hidden_dim, num_class)
        # The linear layer that maps from hidden state space to tag space
        self.batchnorm_rgb = nn.BatchNorm1d(hidden_dim)
        self.batchnorm_flow = nn.BatchNorm1d(hidden_dim)

        # self.batchnorm = nn.BatchNorm1d(2*hidden_dim)
        self.batchsize = batchsize

        self.hidden_rgb = self.init_hidden()
        self.hidden_flow = self.init_hidden()

    def init_hidden(self, single_batchsize=None):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if single_batchsize == None:
            return (torch.zeros(1, self.batchsize, self.hidden_dim).cuda(),
                    torch.zeros(1, self.batchsize, self.hidden_dim).cuda())
        else:
            return (torch.zeros(1, single_batchsize, self.hidden_dim).cuda(),
                    torch.zeros(1, single_batchsize, self.hidden_dim).cuda())

    def clear_history(self, current_batchsize):
        self.hidden_rgb = self.init_hidden(current_batchsize)
        self.hidden_flow = self.init_hidden(current_batchsize)

    def forward(self, sequence_rgb, sequence_flow, labels):
        time_step = sequence_rgb.size(1)
        batchsize = sequence_rgb.size(0)

        sequence_flow = sequence_flow.transpose(0, 1)
        sequence_flow = sequence_flow.float()

        sequence_rgb = sequence_rgb.transpose(0, 1)
        sequence_rgb = sequence_rgb.float()

        rgb_out, self.hidden_rgb = self.lstm_rgb(sequence_rgb, self.hidden_rgb)
        flow_out, self.hidden_flow = self.lstm_flow(sequence_flow, self.hidden_flow)
        # sequence = sequence.mean(0)
        stablize_penalty = stablizing_rnn(rgb_out) + stablizing_rnn(flow_out)

        average_rgb_out = rgb_out.mean(0)
        average_flow_out = flow_out.mean(0)
        average_rgb_out = self.batchnorm_rgb(average_rgb_out)
        average_flow_out = self.batchnorm_flow(average_flow_out)

        average_out = torch.cat([average_rgb_out.transpose(0, 1), average_flow_out.transpose(0, 1)])
        average_out = average_out.transpose(0, 1)

        y_hat, softmax_score, loss = self.memory(average_out, labels, predict=False)

        return y_hat, loss, stablize_penalty

    def predict(self, sequence_rgb, sequence_flow, ):
        sequence_flow = sequence_flow.transpose(0, 1)
        sequence_flow = sequence_flow.float()

        sequence_rgb = sequence_rgb.transpose(0, 1)
        sequence_rgb = sequence_rgb.float()

        rgb_out, self.hidden_rgb = self.lstm_rgb(sequence_rgb, self.hidden_rgb)
        flow_out, self.hidden_flow = self.lstm_flow(sequence_flow, self.hidden_flow)
        # sequence = sequence.mean(0)
        average_rgb_out = rgb_out.mean(0)
        average_flow_out = flow_out.mean(0)
        average_rgb_out = self.batchnorm_rgb(average_rgb_out)
        average_flow_out = self.batchnorm_flow(average_flow_out)
        average_out = torch.cat([average_rgb_out.transpose(0, 1), average_flow_out.transpose(0, 1)])
        average_out = average_out.transpose(0, 1)

        y_hat, softmax_score = self.memory.predict(average_out)
        return y_hat, softmax_score


def stablizing_rnn(hidden_out, beta=500):
    hidden_out = hidden_out.norm(dim=2)

    time_step = hidden_out.size(0)
    batch_size = hidden_out.size(1)
    if hidden_out.is_cuda:
        h_t = torch.zeros(time_step + 1, batch_size).cuda()
        h_t1 = torch.zeros(time_step + 1, batch_size).cuda()
    else:
        h_t = torch.zeros(time_step + 1, batch_size)
        h_t1 = torch.zeros(time_step + 1, batch_size)
    h_t[0:time_step] = hidden_out
    h_t1[1:] = hidden_out
    penalty = (beta / time_step) * (h_t1 - h_t).pow(2).sum()

    return penalty


class LSTM_with_memory(nn.Module):
    def __init__(self, input_dim, hidden_dim, memory_size, num_class, batchsize):
        super(LSTM_with_memory, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm1 = nn.LSTM(input_dim, hidden_dim)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim)
        # The linear layer that maps from hidden state space to tag space
        self.memory = Memory(memory_size, hidden_dim, num_class)
        self.batchsize = batchsize
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.hidden1 = self.init_hidden()
        self.hidden2 = self.init_hidden()

    def init_hidden(self, single_batchsize=None):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if single_batchsize == None:
            return (torch.zeros(1, self.batchsize, self.hidden_dim).cuda(),
                    torch.zeros(1, self.batchsize, self.hidden_dim).cuda())
        else:
            return (torch.zeros(1, single_batchsize, self.hidden_dim).cuda(),
                    torch.zeros(1, single_batchsize, self.hidden_dim).cuda())

    def clear_history(self, current_batchsize):

        self.hidden1 = self.init_hidden(current_batchsize)
        self.hidden2 = self.init_hidden(current_batchsize)

    def forward(self, sequence, labels, predict=False):
        sequence = sequence.float()
        lstm1_out, self.hidden1 = self.lstm1(sequence, self.hidden1)
        # lstm2_out, self.hidden2 = self.lstm2(lstm1_out, self.hidden2)
        average_out = lstm1_out.mean(0)
        average_out = self.batchnorm(average_out)
        average_out = F.relu(average_out)
        y_hat, softmax_score, loss = self.memory(average_out, labels, predict)

        return y_hat, softmax_score, loss

    def predict(self, sequence):
        sequence = sequence.float()
        lstm1_out, self.hidden1 = self.lstm1(sequence, self.hidden1)
        # lstm2_out, self.hidden2 = self.lstm2(lstm1_out, self.hidden2)

        average_out = lstm1_out.mean(0)
        average_out = self.batchnorm(average_out)
        average_out = F.relu(average_out)
        y_hat, softmax_score = self.memory.predict(average_out)

        return y_hat, softmax_score
