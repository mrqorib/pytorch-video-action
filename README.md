# CS5242 - Neural Network and Deep Learning Video Action Classification

Project assignment on AY2019/2020, Semester 2 for video action classification on breakfast actions dataset

## Installation

Project structure and data are from [breakfast actions dataset](https://www.kaggle.com/c/cs5242project/data).

Packages used:
- Python >= 3.6
- PyTorch >= 1.1.0 (recommended: 1.4.0)
- numpy = 1.18.1
- kenlm (optional, to use beam search)
- scikit-learn >= 0.19.1 (optional, to generate new splits)

## Project Structure
/data : I3D data features from Kaggle competition \
/data-comp : Storing features and labels numpy pickle files \
/groundTruth \
&nbsp;&nbsp;&nbsp;&nbsp;/groundTruth: GroundTruth from Kaggle competition \
/models : Storing models save during training \
/results : Storing csv results from the inference \
/splits \
&nbsp;&nbsp;&nbsp;&nbsp;/new_splits : Split done by using data_splitting.py \
&nbsp;&nbsp;&nbsp;&nbsp;/splits : Split from Kaggle competition \
data_splitting.py : Split training dataset into 5 different splits and store the results into /splits/new_splits \
data_utils.py : Data preparation used during training and inference \
inference.py : Infer testing dataset and store results on /results \
inference-scene.py : Infer by aggregating results from same scene and store the results on /results \
networks.py : Storing our models \
train.py : Training network and save each states into /models \
segment.txt : Testing segment from Kaggle competition 

## Usage

### Training

train.py file accepts arguments:

- batchsize
- epoch
- split
- lr
- lr_step_size
- lr_step
- lr_ganma
- num_workers
- models
- pretrained_models
- train_mode
- pred_mode
- load_all
- attn_head
- lstm_layer
- lstm_dropout
- lstm_hidden1
- lstm_hidden2

Example:
```bash
python train.py --batchsize 2 --epoch 20 --split 0 --lr 0.001 --model mstcn --train_mode active --load_all 1
```

### Inference

inference.py and inference-scene.py accept arguments:
- pretrained_model: Accept a list of pretrained_model found on the models/ folder, the naming of the models have to be ${model}_${accuracy}_dev
- load_all: Loading all the data into RAM
- prob: accept 'small' or 'big', this is for aggregation where if majority voting failed and no of frames predicted to be the same, the probability should be taking the higher or lower

Example:
```bash
python inference.py --load_all 1 --prob big --pretrained_model bigru_73.52_dev mstcn_75.59_dev
```
After running the model, the results would be saved in results/ folder with naming such as `result_${model_filename1}_${model_filename2}_${timestamp}.csv` for inference.py and `result_scene_agg_${model_filename1}_${model_filename2}_${timestamp}.csv`

### Add in more networks
1. Go to networks.py to add in class model
2. Go to train.py append in the else-if chain to initialize the network class. Remember that the model name should not have underscore
```bash
elif args.model == 'mstcn':
    net = MultiStageModel(400, n_class=n_class).to(device)
#TODO: add your model name here
# elif args.model == 'mymodel':
#   net = MyNet(<arguments>).to(device)
```
3. Go to inference.py or inference-scene.py append in the else-if chain to initialize the network class
```bash
elif model == 'mstcn':
    net = MultiStageModel(400, n_class=n_class).to(device)
#TODO: add your model name here
# elif args.model == 'mymodel':
#   net = MyNet(<arguments>).to(device)
```

## Kaggle Result
Rank 13 with accuracy 0.74844