import os
from collections import OrderedDict
from random import shuffle

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class BucketBatchSampler(Sampler):
    def __init__(self, inputs, batch_size):
        self.batch_size = batch_size
        ind_n_len = []
        for i, p in enumerate(inputs):
            ind_n_len.append((i, p.shape[0]))
        self.ind_n_len = ind_n_len
        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)

    def _generate_batch_map(self):
        # only to mix the data with same length
        shuffle(self.ind_n_len)
        # sort list by data length
        self.ind_n_len.sort(key=lambda x: x[1])
        # Organize lengths, e.g., batch_map[10] = [30, 124, 203, ...] <= indices of sequences of length 10
        batch_map = OrderedDict()
        for idx, length in self.ind_n_len:
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        batch_list = []
        for length, indices in batch_map.items():
            batch_list += indices
        # make the batches have the same size (as the self.batch_size)
        if len(batch_list) % self.batch_size != 0:
            addition_count = self.batch_size - (len(batch_list) % self.batch_size)
            addition_sample = batch_list[(-2 * addition_count):]
            shuffle(addition_sample)
            batch_list += addition_sample[:addition_count]
        
        group_batch = []
        for i in range(0, len(batch_list), self.batch_size):
            group_batch.append(batch_list[i:i+self.batch_size])
        
        return group_batch

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.ind_n_len)

    def __iter__(self):
        self.group_batch = self._generate_batch_map()
        # shuffle all the batches so they arent ordered by bucket size
        shuffle(self.batch_list)
        for i in self.batch_list:
            yield i
        # for i in range(0, len(self.batch_list), self.batch_size):
        #     yield self.batch_list[i:i+self.batch_size]


class VideoDataset(Dataset):


    def __init__(self, data_dir='./data', annot_path='.', part='train', split=3, load_all=False, mode='active'):
        self.part = part.lower().strip()
        self.split = split
        if self.part not in ["train", "dev", "test"]:
            ValueError("please provide the part only with train/dev/test")
        if part == 'test':
            split_file = os.path.join(annot_path, 'splits', 'splits', '{}.split{}.bundle'.format(part, split))
        else:
            split_file = os.path.join(annot_path, 'splits', 'new_splits', '{}.split{}.bundle'.format(part, split))
        split_content = self._read_file(split_file, offset_start=1)
        self.filenames = self._get_filenames_from_split(split_content)
        
        mapping_file = os.path.join(annot_path, 'splits', 'splits', 'mapping_bf.txt')
        mapping_content = self._read_file(mapping_file)
        self.class_mapping = self._get_class_info(mapping_content)
        
        self.ground_truth_dir = os.path.join(annot_path, 'groundTruth', 'groundTruth')
        self.data_dir = data_dir

        if part == 'test':
            print('Load Segment file')
            segment_file = open('./segment.txt', 'r') 
            segment_lines = segment_file.readlines()
            for index, line in enumerate(segment_lines):
                segment_lines[index] = line.replace('\n', '').split(' ')
            self.segment_lines = segment_lines
        
        self.load_all = load_all
        if self.load_all:
            print('Loading all {} data...'.format(part))
            self._load_all_data()
            print('{} {} instances have been loaded.'.format(len(self.features), part))
        if mode in ['active', 'segment']:
            print('Excluding out SIL frames...')
            self.features, self.labels = self._exclude_label(self.features, self.labels, 0)
        if mode == 'segment':
            print('Converting videos into segments...')
            self._turn_videos_to_segments()
            print('Data has been converted into {} {} segments.'.format(len(self.features), part))


    def _read_file(self, filename, offset_start = 0, offset_end=0):
        with open(filename, 'r') as f:
            file_lines = [x.strip() for x in f.readlines() if len(x.strip()) > 1]
            end_idx = len(file_lines) - offset_end
            return file_lines[offset_start:end_idx]


    def _get_filenames_from_split(self, file_list_raw):
        def process_file_path(file_path_raw):
            return file_path_raw[19:]
        return [process_file_path(file_path) for file_path in file_list_raw]


    def _get_class_info(self, map_list):
        class_info = {
            'class_ids': {},
            'class_names': []
        }
        for pair in map_list:
            line_split = pair.split(' ')
            if len(line_split) < 2:
                continue
            class_idx = int(line_split[0])
            class_name = line_split[1]
            class_info['class_ids'][class_name] = class_idx
            class_info['class_names'].append(class_name)

        return class_info


    def get_class_info(self):
        return self.class_mapping


    def _load_feature_file(self, filename):
        feature_filename = '{}.gz'.format(os.path.splitext(filename)[0])
        feature_file_path = os.path.join(self.data_dir, feature_filename)
        return np.loadtxt(feature_file_path, dtype='float32')


    def _load_label_file(self, filename):
        label_file_path = os.path.join(self.ground_truth_dir, filename)
        str_labels = self._read_file(label_file_path)
        return np.array([self.class_mapping['class_ids'][class_name] \
                            for class_name in str_labels], dtype=np.long)


    def __len__(self):
        return len(self.features or self.filenames)

    
    def _load_all_data(self):
        features_filename = 'data-comp/{}-{}-features.npy'.format(self.part, self.split)
        labels_filename = 'data-comp/{}-{}-labels.npy'.format(self.part, self.split)
        os.makedirs("data-comp", exist_ok=True)
        if self.part == 'test':
            try:
                features = np.load(features_filename, allow_pickle=True)
                print('Pickle files found. Loading from pickles')
            except Exception as e:
                print('Failed loading saved data \n  > ', e)
                print('Loading the data, please wait...')
                features = []
                for filename in self.filenames:
                    feature = self._load_feature_file(filename)
                    features.append(feature)
                try:
                    np.save(features_filename, features)
                    print('All features are successfully saved')
                except Exception as e:
                    print('[WARNING] Failed to save data as pickle\n  > ', e)     
            # slice
            processed_feature = []
            for i, feature in enumerate(features):
                segments = self.segment_lines[i]
                start_frame = int(segments[0])
                end_frame = int(segments[len(segments) - 1])
                processed_feature.append(feature[start_frame : end_frame, :])
                # update self.segment_lines, eg 30 60 70 --> 0 30 40
                self.segment_lines[i] = [int(segment_seq) - int(self.segment_lines[i][0]) for segment_seq in self.segment_lines[i]]
            self.features = processed_feature
            self.labels = None
        else:
            try:
                self.features = np.load(features_filename, allow_pickle=True)
                self.labels = np.load(labels_filename, allow_pickle=True)
                print('Pickle files found. Loading from pickles')
            except Exception as e:
                print('Failed loading saved data \n  > ', e)
                print('Loading the data, please wait...')
                self.features = []
                self.labels = []
                for filename in self.filenames:
                    feature = self._load_feature_file(filename)
                    label = self._load_label_file(filename)
                    self.features.append(feature)
                    self.labels.append(label)
                try:
                    np.save(features_filename, self.features)
                    np.save(labels_filename, self.labels)
                    print('All features and labels are successfully saved')
                except Exception as e:
                    print('[WARNING] Failed to save data as pickle\n  > ', e)


    def _exclude_label(self, data_feat, data_labels, label):
        """Exclude specified label from data_feat and data_labels
            # Arguments
                data_feat: a list of 2D tensor (no_frame, feat)
                data_labels: 2D list 
                label: a string or number, e.g. 0, 1, 2...
            # Returns
                data_feat_result: a list of 2D tensor (no_frame, feat)
                data_labels_result: 2D list
        """
        data_feat_result = []
        data_labels_result = []
        for iter_index, file_content in enumerate(data_labels):
            indexes = [i for i,x in enumerate(file_content) if str(x) == str(label)]
            data_labels_result.append(np.delete(np.array(file_content), indexes))
            data_feat_result.append(np.delete(np.array(data_feat[iter_index]), indexes, axis=0))
        return np.array(data_feat_result), np.array(data_labels_result)


    def _turn_videos_to_segments(self):
        segments = []
        labels = []
        for i, video in enumerate(self.features):
            label = self.labels[i]
            segment, label, length = self.get_label_length_seq(video, label)
            segments += segment
            labels += label
        self.features = np.array(segments)
        self.labels = np.array(labels)


    def get_label_length_seq(self, frame, label):
        frame_seq = []
        label_seq = []
        length_seq = []
        start = 0
        length_seq.append(0)
        for i in range(len(label)):
            if label[i] != label[start]:
                label_seq.append(label[start])
                frame_seq.append(frame[start:i,:])
                length_seq.append(i)
                start = i
        label_seq.append(label[start])
        frame_seq.append(frame[start:,:])
        length_seq.append(len(frame))

        return frame_seq, label_seq, length_seq


    def _get_feature(self, idx):
        if self.load_all:
            return self.features[idx]
        else:
            return self._load_feature_file(self.filenames[idx])


    def _get_label(self, idx):
        if self.load_all:
            return self.labels[idx]
        else:
            return self._load_label_file(self.filenames[idx])


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data = torch.tensor(self._get_feature(idx))
        if self.part == 'test':
            label = []
        else:
            label = self._get_label(idx)
        label = torch.tensor(label, dtype=torch.long)
        # label = label.type(torch.long)
        return (data, label)
