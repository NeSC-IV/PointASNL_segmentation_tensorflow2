#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import yaml
import random
import semantic_label

splits = ['train', 'valid', 'test']
mapped_content = {0: 0.03150183342534689, 1: 0.042607828674502385, 2: 0.00016609538710764618, 3: 0.00039838616015114444,
                  4: 0.0021649398241338114, 5: 0.0018070552978863615, 6: 0.0003375832743104974,
                  7: 0.00012711105887399155, 8: 3.746106399997359e-05, 9: 0.19879647126983288, 10: 0.014717169549888214,
                  11: 0.14392298360372, 12: 0.0039048553037472045, 13: 0.1326861944777486, 14: 0.0723592229456223,
                  15: 0.26681502148037506, 16: 0.006035012012626033, 17: 0.07814222006271769, 18: 0.002855498193863172,
                  19: 0.0006155958086189918}

seed = 100


# In[2]:


class Semantickitti():
    def __init__(self, root, sample_points=8192, block_size=10, num_classes=20, split='train', with_remission=False, config_file='./semantic-kitti.yaml', should_map=True, padding=0.01, random_sample=False, random_rate=0.1):
        self.root = root
        assert split in splits
        self.split = split
        self.padding = padding
        self.block_size = block_size
        self.sample_points = sample_points
        self.random_sample = random_sample
        self.with_remission = with_remission
        self.should_map = should_map
        self.config = yaml.safe_load(open(config_file, 'r'))
        self.scan = semantic_label.SemLaserScan(
            nclasses=num_classes, sem_color_dict=self.config['color_map'])
        sequences = self.config['split'][split]
        self.points_name = []
        self.label_name = []
        for sequence in sequences:
            sequence = '{0:02d}'.format(int(sequence))
            points_path = os.path.join(
                self.root, 'sequences', sequence, 'velodyne')
            label_path = os.path.join(
                self.root, 'sequences', sequence, 'labels')
            seq_points_name = [os.path.join(points_path, pn) for pn in os.listdir(
                points_path) if pn.endswith('.bin')]
            seq_label_name = [os.path.join(label_path, ln) for ln in os.listdir(
                label_path) if ln.endswith('.label')]
            assert len(seq_points_name) == len(seq_label_name)
            seq_points_name.sort()
            seq_label_name.sort()
            self.points_name.extend(seq_points_name)
            self.label_name.extend(seq_label_name)

        if self.random_sample:
            random.Random(seed).shuffle(self.points_name)
            random.Random(seed).shuffle(self.label_name)
            total_length = len(self.points_name)
            self.points_name = self.points_name[:int(total_length*random_rate)]
            self.label_name = self.label_name[:int(total_length*random_rate)]

        label_weights_dict = mapped_content
        num_keys = len(label_weights_dict.keys())
        self.label_weights_table = np.zeros((num_keys), dtype=np.float32)
        self.label_weights_table[list(label_weights_dict.keys())] = list(
            label_weights_dict.values())
        self.label_weights_table = np.power(
            np.max(self.label_weights_table[1:])/self.label_weights_table, 1/3.0)

        if should_map:
            mapdict = self.config['learning_map']
            maxkey = max(mapdict.keys())
            self.map_table = np.zeros((maxkey+100), dtype=np.int32)
            self.map_table[list(mapdict.keys())] = list(mapdict.values())

    def __getitem__(self, index):
        points_name, label_name = self.points_name[index], self.label_name[index]
        self.scan.open_scan(points_name)
        self.scan.open_label(label_name)
        points = self.scan.points
        label = self.scan.sem_label
        if self.should_map:
            label = self.map_table[label]
        label_weights = self.label_weights_table[label]
        coordmax = np.max(points[:, 0:3], axis=0)
        coordmin = np.min(points[:, 0:3], axis=0)

        for i in range(10):
            curcenter = points[np.random.choice(len(label), 1)[0], 0:3]
            curmin = curcenter-[self.block_size/2, self.block_size/2, 14]
            curmax = curcenter+[self.block_size/2, self.block_size/2, 14]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum(
                (points[:, 0:3] >= (curmin-0.2))*(points[:, 0:3] <= (curmax+0.2)), axis=1) == 3
            cur_point_set = points[curchoice, 0:3]
            cur_point_full = points[curchoice, :]
            cur_semantic_seg = label[curchoice]
            if len(cur_semantic_seg) == 0:
                continue
            mask = np.sum((cur_point_set >= (curmin-self.padding))
                          * (cur_point_set <= (curmax+self.padding)), axis=1) == 3
            isvalid = np.sum(cur_semantic_seg > 0)/len(cur_semantic_seg) >= 0.7
            if isvalid:
                break

        choice = np.random.choice(
            len(cur_semantic_seg), self.sample_points, replace=True)
        point_set = cur_point_full[choice, :]
        semantic_seg = cur_semantic_seg[choice]
        mask = mask[choice]
        sample_weight = label_weights[semantic_seg]
        sample_weight *= mask
        if self.with_remission:
            point_set = np.concatenate((point_set, np.expand_dims(
                self.scan.remissions[choice], axis=1)), axis=1)

        return point_set, semantic_seg, sample_weight

    def __len__(self):
        return len(self.points_name)


class semantickittidataset_whole():
    def __init__(self, root, sample_points=8192, block_size=10, num_classes=20, split='train', with_remission=False, config_file='./semantic-kitti.yaml', should_map=True, padding=0.01, random_sample=False, random_rate=0.1):
        self.root = root
        assert split in splits
        self.split = split
        self.padding = padding
        self.block_size = block_size
        self.sample_points = sample_points
        self.random_sample = random_sample
        self.with_remission = with_remission
        self.should_map = should_map
        self.config = yaml.safe_load(open(config_file, 'r'))
        self.scan = semantic_label.SemLaserScan(
            nclasses=num_classes, sem_color_dict=self.config['color_map'])
        sequences = self.config['split'][split]

        self.points_name = []
        self.label_name = []

        for sequence in sequences:
            sequence = '{0:02d}'.format(int(sequence))
            points_path = os.path.join(
                self.root, 'sequences', sequence, 'velodyne')
            label_path = os.path.join(
                self.root, 'sequences', sequence, 'labels')
            seq_points_name = [os.path.join(points_path, pn) for pn in os.listdir(
                points_path) if pn.endswith('.bin')]
            seq_label_name = [os.path.join(label_path, ln) for ln in os.listdir(
                label_path) if ln.endswith('.label')]
            assert len(seq_points_name) == len(seq_label_name)
            seq_label_name.sort()
            seq_points_name.sort()
            self.points_name.extend(seq_points_name)
            self.label_name.extend(seq_label_name)
        if self.random_sample:
            random.Random(seed).shuffle(self.points_name)
            random.Random(seed).shuffle(self.label_name)
            total_length = len(self.points_name)
            self.points_name = self.points_name[:int(total_length*random_rate)]
            self.label_name = self.label_name[:int(total_length*random_rate)]
        label_weights_dict = mapped_content
        num_keys = len(label_weights_dict.keys())
        self.label_weights_table = np.zeros((num_keys), dtype=np.float32)
        self.label_weights_table[list(label_weights_dict.keys())] = list(
            label_weights_dict.values())
        self.label_weights_table = np.power(
            np.max(self.label_weights_table[1:])/self.label_weights_table, 1/3.0)
        if should_map:
            remapdict = self.config["learning_map"]
            maxkey = max(remapdict.keys())
            self.remap_tabel = np.zeros((maxkey+100), dtype=np.int32)
            self.remap_tabel[list(remapdict.keys())] = list(remapdict.values())

    def __getitem__(self, index):
        points_name, label_name = self.points_name[index], self.label_name[index]
        self.scan.open_scan(points_name)
        self.scan.open_label(label_name)
        points = self.scan.points
        label = self.scan.sem_label
        if self.should_map:
            label = self.remap_tabel[label]
        label_weights = self.label_weights_table[label]
        coordmax = np.max(points[:, 0:3], axis=0)
        coordmin = np.min(points[:, 0:3], axis=0)

        nsubvolume_x = np.ceil(
            (coordmax[0]-coordmin[0])/self.block_size).astype(np.int32)
        nsubvolume_y = np.ceil(
            (coordmax[1]-coordmin[1])/self.block_size).astype(np.int32)
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin+[i*self.block_size, j*self.block_size, 0]
                curmax = coordmin+[(i+1)*self.block_size, (j+1)
                                   * self.block_size, coordmax[2]-coordmin[2]]
                curchoice = np.sum(
                    (points[:, 0:3] >= (curmin-0.2))*(points[:, 0:3] <= (curmax+0.2)), axis=1) == 3
                cur_point_set = points[curchoice, 0:3]
                cur_point_full = points[curchoice, :]
                cur_semantic_seg = label[curchoice]
                if len(cur_semantic_seg) == 0:
                    continue
                mask = np.sum((cur_point_set >= (curmin-self.padding))
                              * (cur_point_set <= (curmax+self.padding)), axis=1) == 3
                choice = np.random.choice(
                    len(cur_semantic_seg), self.sample_points, replace=True)
                point_set = cur_point_full[choice, :]
                if self.with_remission:
                    point_set = np.concatenate((point_set, np.expand_dims(
                        self.scan.remissions[choice], axi=1)), axis=1)
                semantic_seg = cur_semantic_seg[choice]
                mask = mask[choice]
                sample_weight = label_weights[semantic_seg]
                sample_weight *= mask
                point_sets.append(np.expand_dims(point_set, 0))
                semantic_segs.append(np.expand_dims(semantic_seg, 0))
                sample_weights.append(np.expand_dims(sample_weight, 0))
        point_sets = np.concatenate(tuple(point_sets), axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs), axis=0)
        sample_weights = np.concatenate(tuple(sample_weights), axis=0)

        return point_sets, semantic_segs, sample_weights

    def __len__(self):
        return len(self.points_name)


# In[3]:


def batch_data(dataset, idxs, start_idx, end_idx, num_point, feature_channel):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, num_point, 3+feature_channel))
    batch_label = np.zeros((bsize, num_point), dtype=np.int32)
    batch_smpw = np.zeros((bsize, num_point), dtype=np.float32)
    for i in range(bsize):
        ps, seg, smpw = dataset[idxs[i+start_idx]]
        batch_data[i, ...] = ps
        batch_label[i, :] = seg
        batch_smpw[i, :] = smpw
    return batch_data, batch_label, batch_smpw
