import os
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):


    # TODO: Fix load_all
    def __init__(self, data_dir='./data', annot_path='.', part='train', load_all=False):
        self.part = part.lower().strip()
        if self.part not in ["train", "test"]:
            ValueError("please provide the part only with train/test")
        split_file = os.path.join(annot_path, 'splits', 'splits', '{}.split1.bundle'.format(part))
        split_content = self._read_file(split_file, offset_start=1)
        self.filenames = self._get_filenames_from_split(split_content)
        
        mapping_file = os.path.join(annot_path, 'splits', 'splits', 'mapping_bf.txt')
        mapping_content = self._read_file(mapping_file)
        self.class_mapping = self._get_class_info(mapping_content)
        
        self.ground_truth_dir = os.path.join(annot_path, 'groundTruth', 'groundTruth')
        self.data_dir = data_dir

        self.load_all = load_all
        if self.load_all:
            self._load_all_data()


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
        return len(self.filenames)

    
    def _load_all_data(self):
        self.features = []
        self.labels = []
        for filename in self.filenames:
            feature = self._load_feature_file(filename)
            label = self._load_label_file(filename)
            self.features.append(feature)
            self.labels.append(label)


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
        
        data = torch.from_numpy(self._get_feature(idx))
        if self.part == 'test':
            label = []
        else:
            label = self._get_label(idx)
        label = torch.from_numpy(label)
        label = label.type(torch.long)
        return (data, label)
