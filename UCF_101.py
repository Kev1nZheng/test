import re
import os
import pandas as pd
import torch.utils.data as data
import numpy as np


class UCF101_Dataset(data.Dataset):
    def __init__(self, root_dir, frame_length, endwith):
        self.root_dir = root_dir
        self.categories = sorted(os.listdir(root_dir))
        self.cat2idx = dict(zip(self.categories, range(len(self.categories))))
        self.idx2cat = dict(zip(self.cat2idx.values(), self.cat2idx.keys()))
        self.files = []
        self.labels = []
        self.pathes = []
        self.endwith = endwith
        self.frame_length = frame_length
        for (dirpath, dirnames, filenames) in os.walk(self.root_dir):
            dirnames.sort()
            # sort_nicely(filenames)
            for f in filenames:
                if f.endswith(self.endwith):
                    o = {}
                    o['feature_path'] = dirpath + '/' + f
                    # print(os.path.basename(dirpath))
                    o['category'] = self.cat2idx[os.path.basename(dirpath)]
                    self.files.append(o)
                    self.labels.append(self.cat2idx[os.path.basename(dirpath)])
                    self.pathes.append(dirpath + '/' + f)

    def __getitem__(self, idx):
        feature_path = self.files[idx]['feature_path']
        category = self.files[idx]['category']
        # feature = pd.read_csv(feature_path, sep=",", header=None)
        feature = np.load(feature_path)
        feature = np.asarray(feature)
        if feature.shape[0] >= self.frame_length:
            feature = feature[:self.frame_length, :]
        else:
            pad_size = self.frame_length - feature.shape[0]
            feature = np.pad(feature, [(0, pad_size), (0, 0)], 'constant', constant_values=(0))
        # feature = np.mean(feature, axis=0)
        # print(feature.shape)
        # feature = feature.transpose()
        return feature, category

    def __len__(self):
        return len(self.files)


class UCF101_Dataset_Clips(data.Dataset):
    def __init__(self, root_dir, clip, endwith):
        self.root_dir = root_dir
        self.categories = sorted(os.listdir(root_dir))
        self.cat2idx = dict(zip(self.categories, range(len(self.categories))))
        self.idx2cat = dict(zip(self.cat2idx.values(), self.cat2idx.keys()))
        self.files = []
        self.labels = []
        self.pathes = []
        self.endwith = endwith
        self.clip = clip
        for (dirpath, dirnames, filenames) in os.walk(self.root_dir):
            dirnames.sort()
            # sort_nicely(filenames)
            for f in filenames:
                if f.endswith(self.endwith):
                    o = {}
                    o['feature_path'] = dirpath + '/' + f
                    # print(os.path.basename(dirpath))
                    o['category'] = self.cat2idx[os.path.basename(dirpath)]
                    self.files.append(o)
                    self.labels.append(self.cat2idx[os.path.basename(dirpath)])
                    self.pathes.append(dirpath + '/' + f)

    def __getitem__(self, idx):
        feature_path = self.files[idx]['feature_path']
        category = self.files[idx]['category']
        # feature = pd.read_csv(feature_path, sep=",", header=None)
        feature = np.load(feature_path)
        feature = np.asarray(feature)
        frames_per_clip = feature.shape[0] // 10
        frame_end = frames_per_clip * self.clip
        feature = feature[0:frame_end, :]
        # print(feature_path,frame_star,frame_end)
        # feature = np.mean(feature, axis=0)
        # print(feature.shape)
        # feature = feature.transpose()
        return feature, category

    def __len__(self):
        return len(self.files)


class UCF101_fusion(data.Dataset):
    def __init__(self, root_dir_1, root_dir_2, frame_length, endwith):
        self.root_dir_1 = root_dir_1
        self.root_dir_2 = root_dir_2
        self.categories = sorted(os.listdir(root_dir_1))
        self.cat2idx = dict(zip(self.categories, range(len(self.categories))))
        self.idx2cat = dict(zip(self.cat2idx.values(), self.cat2idx.keys()))
        self.files_1 = []
        self.files_2 = []
        self.labels = []
        self.pathes_1 = []
        self.pathes_2 = []
        self.endwith = endwith
        self.frame_length = frame_length
        for (dirpath, dirnames, filenames) in os.walk(self.root_dir_1):
            dirnames.sort()
            sort_nicely(filenames)
            for f in filenames:
                if f.endswith(self.endwith):
                    o = {}
                    o['feature_path'] = dirpath + '/' + f
                    # print(os.path.basename(dirpath))
                    o['category'] = self.cat2idx[os.path.basename(dirpath)]
                    self.files_1.append(o)
                    self.labels.append(self.cat2idx[os.path.basename(dirpath)])
                    self.pathes_1.append(dirpath + '/' + f)

        for (dirpath, dirnames, filenames) in os.walk(self.root_dir_2):
            dirnames.sort()
            sort_nicely(filenames)
            for f in filenames:
                if f.endswith(self.endwith):
                    o = {}
                    o['feature_path'] = dirpath + '/' + f
                    # print(os.path.basename(dirpath))
                    o['category'] = self.cat2idx[os.path.basename(dirpath)]
                    self.files_2.append(o)
                    self.pathes_2.append(dirpath + '/' + f)

    def __getitem__(self, idx):
        feature_path_rgb = self.files_1[idx]['feature_path']
        feature_path_flow = self.files_2[idx]['feature_path']
        category = self.files_1[idx]['category']
        # feature = pd.read_csv(feature_path, sep=",", header=None)
        feature_rgb = np.load(feature_path_rgb)
        feature_rgb = np.asarray(feature_rgb)
        if feature_rgb.shape[0] >= self.frame_length:
            feature_rgb = feature_rgb[:self.frame_length, :]
        else:
            pad_size = self.frame_length - feature_rgb.shape[0]
            feature_rgb = np.pad(feature_rgb, [(0, pad_size), (0, 0)], 'constant', constant_values=(0))

        feature_flow = np.load(feature_path_flow)
        feature_flow = np.asarray(feature_flow)
        if feature_flow.shape[0] >= self.frame_length:
            feature_flow = feature_flow[:self.frame_length, :]
        else:
            pad_size = self.frame_length - feature_flow.shape[0]
            feature_flow = np.pad(feature_flow, [(0, pad_size), (0, 0)], 'constant', constant_values=(0))
        # feature = np.mean(feature, axis=0)
        # print(feature.shape)
        # feature = feature.transpose()
        return feature_rgb, feature_flow, category

    def __len__(self):
        return len(self.files_1)


class UCF101_fusion_Clips(data.Dataset):
    def __init__(self, root_dir_1, root_dir_2, clip, endwith):
        self.root_dir_1 = root_dir_1
        self.root_dir_2 = root_dir_2
        self.categories = sorted(os.listdir(root_dir_1))
        self.cat2idx = dict(zip(self.categories, range(len(self.categories))))
        self.idx2cat = dict(zip(self.cat2idx.values(), self.cat2idx.keys()))
        self.files_1 = []
        self.files_2 = []
        self.labels = []
        self.pathes_1 = []
        self.pathes_2 = []
        self.endwith = endwith
        self.clip = clip
        for (dirpath, dirnames, filenames) in os.walk(self.root_dir_1):
            dirnames.sort()
            sort_nicely(filenames)
            for f in filenames:
                if f.endswith(self.endwith):
                    o = {}
                    o['feature_path'] = dirpath + '/' + f
                    # print(os.path.basename(dirpath))
                    o['category'] = self.cat2idx[os.path.basename(dirpath)]
                    self.files_1.append(o)
                    self.labels.append(self.cat2idx[os.path.basename(dirpath)])
                    self.pathes_1.append(dirpath + '/' + f)

        for (dirpath, dirnames, filenames) in os.walk(self.root_dir_2):
            dirnames.sort()
            sort_nicely(filenames)
            for f in filenames:
                if f.endswith(self.endwith):
                    o = {}
                    o['feature_path'] = dirpath + '/' + f
                    # print(os.path.basename(dirpath))
                    o['category'] = self.cat2idx[os.path.basename(dirpath)]
                    self.files_2.append(o)
                    self.pathes_2.append(dirpath + '/' + f)

    def __getitem__(self, idx):
        feature_path_rgb = self.files_1[idx]['feature_path']
        feature_path_flow = self.files_2[idx]['feature_path']
        category = self.files_1[idx]['category']
        # feature = pd.read_csv(feature_path, sep=",", header=None)
        feature_rgb = np.load(feature_path_rgb)
        feature_rgb = np.asarray(feature_rgb)

        rgb_frames_per_clip = feature_rgb.shape[0] // 10
        rgb_frame_end = rgb_frames_per_clip * self.clip
        feature_rgb = feature_rgb[0:rgb_frame_end, :]

        feature_flow = np.load(feature_path_flow)
        feature_flow = np.asarray(feature_flow)

        flow_frames_per_clip = feature_flow.shape[0] // 10
        flow_frames_end = flow_frames_per_clip * self.clip
        feature_flow = feature_flow[:flow_frames_end, :]
        # feature = np.mean(feature, axis=0)
        # print(feature.shape)
        # feature = feature.transpose()
        return feature_rgb, feature_flow, category

    def __len__(self):
        return len(self.files_1)


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def test():
    root_dir = '/home/shangqian/workspace/LSTM_Memory/Data/RGB_Features_npy/'
    end_with = 'npy'
    test_dataset = UCF101_Dataset(root_dir, end_with)
    print(test_dataset.labels[1000])
    item1 = test_dataset.__getitem__(1000)
    print(item1[0].shape)
    print(item1[1])

    item2 = test_dataset.__getitem__(1)
    print(item2[0].shape)
    print(item2[1])
    batchsize = 4
    dataloader = data.DataLoader(test_dataset, batch_size=batchsize,
                                 shuffle=True, num_workers=2)

    x_batch, y_batch = next(iter(dataloader))
    print(y_batch.size(0))

# test()
