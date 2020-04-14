import os


def read_file(filename, offset_start = 0, offset_end=0):
    with open(filename, 'r') as f:
        file_lines = [x.strip() for x in f.readlines() if len(x.strip()) > 1]
        end_idx = len(file_lines) - offset_end
        return file_lines[offset_start:end_idx]


def get_filenames_from_split(file_list_raw):
    def process_file_path(file_path_raw):
        return file_path_raw[19:]
    return [process_file_path(file_path) for file_path in file_list_raw]


def load_label_file(ground_truth_dir, class_mapping, filename):
    label_file_path = os.path.join(ground_truth_dir, filename)
    str_labels = read_file(label_file_path)
    return get_label_length_seq([class_mapping['class_ids'][class_name] \
                                 for class_name in str_labels])


def get_label_length_seq(label):
    label_seq = []
    start = 0
    for i in range(len(label)):
        if label[i] != label[start]:
            label_seq.append(label[start])
            start = i
    label_seq.append(label[start])

    return label_seq

def get_class_info(map_list):
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


def main():
    annot_path='.'
    split_file = os.path.join(annot_path, 'splits', 'splits', 'train.split1.bundle')
    ground_truth_dir = os.path.join(annot_path, 'groundTruth', 'groundTruth')
    split_content = read_file(split_file, offset_start=1)
    filenames = get_filenames_from_split(split_content)

    mapping_file = os.path.join(annot_path, 'splits', 'splits', 'mapping_bf.txt')
    mapping_content = read_file(mapping_file)
    class_mapping = get_class_info(mapping_content)

    f = open(os.path.join(annot_path, 'groundTruth', 'segment_labels.txt'), 'w')
    for filename in filenames:
        label = load_label_file(ground_truth_dir, class_mapping, filename)
        f.write(' '.join(label) + '\n')
    f.close()


if __name__ == "__main__":
    main()