import os
import collections
import numpy as np
from sklearn.model_selection import StratifiedKFold

# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
# >>> y = np.array([0, 0, 1, 1])
# >>> skf = StratifiedKFold(n_splits=2)
# >>> skf.get_n_splits(X, y)
# 2
# >>> print(skf)
# StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
# >>> for train_index, test_index in skf.split(X, y):

def get_class_info(map_list):
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

def read_file(filename, offset_start = 0, offset_end=0):
    with open(filename, 'r') as f:
        file_lines = [x.strip() for x in f.readlines() if len(x.strip()) > 1]
        end_idx = len(file_lines) - offset_end
        return file_lines[offset_start:end_idx]

def process_file_path(file_path_raw):
    return os.path.splitext(file_path_raw[19:])[0]

def get_filenames_from_split(file_list_raw):
    return [process_file_path(file_path) for file_path in file_list_raw]

def load_label_file(filename, ground_truth_dir, class_mapping):
        label_file_path = os.path.join(ground_truth_dir, filename)
        str_labels = read_file(label_file_path)
        return np.array([class_mapping['class_ids'][class_name] \
                            for class_name in str_labels], dtype=np.long)

def main():
    data_dir = './data'
    annot_path = '.'
    split_file = os.path.join(annot_path, 'splits', 'splits', 'train.split1.bundle')
    split_content = read_file(split_file, offset_start=1)
    
    mapping_file = os.path.join(annot_path, 'splits', 'splits', 'mapping_bf.txt')
    mapping_content = read_file(mapping_file)
    class_mapping = get_class_info(mapping_content)

    ground_truth_dir = os.path.join(annot_path, 'groundTruth', 'groundTruth')

    action_ids = {}
    x = []
    y = []
    c = []
    for filepath in split_content:
        filename = process_file_path(filepath)
        parts = filename.split('_')
        action = parts[-1]
        camera_type = parts[1]
        if action not in action_ids:
            action_ids[action] = len(action_ids)
        
        x.append(filepath)
        y.append(action_ids[action])
        c.append(camera_type)

    skf = StratifiedKFold(n_splits=5, random_state=123)
    x = np.array(x)
    y = np.array(y)
    c = np.array(c)
    
    print(action_ids)
    
    for part_idx, (train_index, dev_index) in enumerate(skf.split(x, y)):
        x_train = x[train_index]
        y_train = y[train_index]
        c_train = c[train_index]
        x_dev = x[dev_index]
        y_dev = y[dev_index]
        c_dev = c[dev_index]
        train_action = collections.Counter(y_train)
        dev_action = collections.Counter(y_dev)

        # class_occ = {}
        # for x in x_train:
        #     filename = process_file_path(filepath)
        #     label = load_label_file(filename)

        #     current_class_occ = collections.Counter(label)
        #     for key, value in current_class_occ.items():
        #         class_occ[key] = value + class_occ.get(key, 0)
        
        print('Partition ', part_idx)
        print('Train action ', str(train_action))
        print('Dev action ', str(dev_action))
        print('Train cameras ', str(collections.Counter(c_train)))
        print('Dev cameras ', str(collections.Counter(c_dev)))

        train_filename = os.path.join(annot_path, 'splits', 'new_splits',
                                        'train.split{}.bundle'.format(part_idx))
        with open(train_filename, 'w') as f:
            f.write('# ' + str(train_action))
            for filepath in x_train:
                f.write(filepath + '\n')
        
        dev_filename = os.path.join(annot_path, 'splits', 'new_splits',
                                        'dev.split{}.bundle'.format(part_idx))
        with open(dev_filename, 'w') as f:
            f.write('# ' + str(dev_action))
            for filepath in x_dev:
                f.write(filepath + '\n')




if __name__ == "__main__":
    main()