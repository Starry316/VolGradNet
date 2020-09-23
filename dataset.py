import glob
import numpy as np
import torch
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, training=True, dir='data/', feature_num=7):
        self.training = training
        self.feature_num = feature_num
        self.data_dir = dir

        if self.training:
            self.data_dir = self.data_dir + 'train/'
            print('training set')
        else:
            self.data_dir = self.data_dir + 'test/'
            print('testing set')

        self.file_list = glob.glob(self.data_dir + 'image/' + 'v*.npy')
        print(len(self.file_list))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        fn = self.file_list[index].split('\\')[-1].split('.npy')[0]
        i = np.load(self.data_dir + 'image/' + fn + '.npy')[0]
        i = torch.from_numpy(i)
        dx = np.load(self.data_dir  + 'grad/' + fn + '-gx.npy')[0]
        dy = np.load(self.data_dir  + 'grad/' + fn + '-gy.npy')[0]
        dx = torch.from_numpy(dx)
        dy = torch.from_numpy(dy)
        feature = np.load(self.data_dir + 'feature/' + fn + '-feature.npy')[0]
        feature = feature[:self.feature_num, :, :]
        feature = torch.from_numpy(feature)
        recon = np.load(self.data_dir  + 'recon/' + fn + '-recon.npy')[0]
        recon = torch.from_numpy(recon)
        return i, dx, dy, feature, recon


# class DataPrefetcher():
#     “”“
#     Data prefetcher to speed up IO
#     “”“
#     def __init__(self, loader):
#         self.loader = iter(loader)
#         self.stream = torch.cuda.Stream()
#         self.preload()
#
#     def preload(self):
#         try:
#             self.next_i,self.next_dx, self.next_dy, self.next_f,  self.next_recon = next(self.loader)
#         except StopIteration:
#             self.next_i = None
#             self.next_dx = None
#             self.next_dy = None
#             self.next_f = None
#             self.next_recon = None
#             return
#         with torch.cuda.stream(self.stream):
#             self.next_i = self.next_i.cuda(non_blocking=True)
#
#             self.next_dx = self.next_dx.cuda(non_blocking=True)
#             self.next_dy = self.next_dy.cuda(non_blocking=True)
#             self.next_f = self.next_f.cuda(non_blocking=True)
#             self.next_recon = self.next_recon.cuda(non_blocking=True)
#
#     def next(self):
#         torch.cuda.current_stream().wait_stream(self.stream)
#         i = self.next_i
#         dx = self.next_dx
#         dy = self.next_dy
#         f = self.next_f
#         recon = self.next_recon
#         self.preload()
#         return i, dx, dy, f, recon