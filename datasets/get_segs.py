import numpy as np
import os
import torch
import torch.utils.data as data
import json
import cv2

class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 datafile,
                 image_transform,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=False,
                 seg_part = 0):
        self.npoints = npoints
        self.datafile = datafile
        self.catfile = './datasets/synsetoffset2category.txt'
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.datafile, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.datafile, category, 'points', uuid + '.pts'),
                        os.path.join(self.datafile, category, 'points_label',uuid + '.seg'),
                        os.path.join(self.datafile, category, 'image_renders', uuid, 'rendering')))#, '{02d}.png'.format(i)) for i in range(24)))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                for j in range(24):
                    f = os.path.join(fn[2], '{:02d}.png'.format(j))
                    self.datapath.append((item, fn[0], fn[1], f))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        with open('./datasets/num_seg_classes.txt', 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        #print(self.seg_classes, self.num_seg_classes)
        self.image_transform = image_transform
    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        #image = np.load(fn[3])
        image = np.transpose(np.array(cv2.imread(fn[3]), dtype = np.uint8), (2, 0, 1))
        if self.image_transform is not None:
            image = self.image_transform(image)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale


        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter
        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        image = torch.from_numpy(image)

        return point_set, seg, image, self.num_seg_classes

    def __len__(self):
        return len(self.datapath)


if __name__=='__main__':
    datapath='/Users/lmy/Dataset/Shapenet/shapenetcore_partanno_segmentation_benchmark_v0/'
    d = ShapeNetDataset(datafile=datapath, class_choice=['Chair'])
    print(len(d))
    ps, seg, image = d[0]
    print(ps.size(), ps.type(), seg.size(), seg.type())
    print(image.size(), image.type())






