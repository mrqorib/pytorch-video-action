import os
import torch
from torch.utils.data import Dataset, DataLoader


class VideoDataset(Dataset):


    def __init__(self, data_dir='./data', annot_path='.', part='train'):
        part = part.lower().strip()
        if part not in ["train", "test"]:
            ValueError("please provide the part only with train/test")
        split_file = os.path.join(annot_path, 'splits', 'splits', '{}.split1.bundle'.format(part))
        split_content = self._read_file(split_file)
        self.filenames = self._get_filenames_from_split(split_content)
        
        mapping_file = os.path.join(annot_path, 'splits', 'splits', 'mapping_bf.txt')
        mapping_content = self._read_file(mapping_file)
        self.class_mapping = self._get_class_info(mapping_content)
        
        self.ground_truth_dir = os.path.join(annot_path, 'groundTruth', 'groundTruth')
        self.data_dir = data_dir


    def _read_file(self, filename, offset_start = 0, offset_end=0):
        with open(filename, 'r') as f:
            file_lines = [x for x in f.readlines() if len(x) > 1]
            end_idx = len(file_lines) - offset_end
            return file_lines[offset_start:end_idx]


    def _get_filenames_from_split(self, file_list_raw):
        def process_file_path(file_path_raw):
            return file_list_raw[19:]
        return process_file_path(file_list_raw)


    def _get_class_info(self, map_list):
        class_info = {
            'class_ids': {},
            'class_names': []
        }
        for pair in map_list:
            line_split = pair.split(' ')
            if len(line_split) < 2:
                continue
            class_idx = line_split[0]
            class_name = line_split[1]
            class_info['class_ids'][class_name] = class_idx
            class_info['class_names'].append(class_name)

        return class_info


    def _get_feature(self, filename):
        feature_filename = '{}.gz'.format(os.path.splitext(filename)[0])
        feature_file_path = os.path.join(self.data_dir, feature_filename)
        return np.loadtxt(feature_file_path, dtype='float32')


    def _get_label(self, filename):
        label_file_path = os.path.join(self.ground_truth_dir, filename)
        return self._read_file(label_file_path)


    def __len__(self):
        return len(self.filenames)

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data = self._get_feature(self.filenames[idx])
        label = self._get_label(self.filenames[idx])
        return (data, label)
