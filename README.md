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

There are three types of training mode:
- Active (default)
In this training scheme, the SIL frames are excluded
- Cont
This mode trains the whole video contiguously, as real-world application
- Segment
This mode will pass the training instance as the segments of video instead of the whole video

Other than the data-processing, the script also accept 3 types of prediction mode:
- Cont (default)
Predict the class for each frame. Use this for cont, active, and segment training mode
- last
Only use the output of the last time-step for the prediction. Use this for the segment training mode
- avg
Average the output of each time-step for the prediction. Use this for the segment training mode

train.py file accepts arguments:
| Arguments          | Type  | Default | Remarks
| -------------------|-------|---------|---------
| batchsize          | Int   | 1       | Batch size, default: 1
| epoch              | Int   | 10      | Number of training epoch
| split              | Int   | 0       | Dataset split, options: [0-4]
| lr                 | Float | 0.001   | Initial learning rate
| lr_step_size       | Int   | -       | Number of epoch to change the learning rate (lr) into lr = lr * lr_gamma
| lr_gamma           | Float | -       | Multiplier of learning rate every lr_step_size epoch
| num_workers        | Int   | 0       | Number of workers to load dataset. For windows user, keep it 0 to prevent errors
| models             | String| -       | Choose which model to train, options: ['vanillalstm', 'bilstm', 'bigru', 'mstcn']. There are other experimental options also (not recommended): ['bilstmlm', 'attn', 'winattn', 'ctcloss']
| pretrained_models  | String| -       | name of the model inside the ./model folder (without the .pth extension)
| train_mode         | String| active  | options: ['segment', 'active', 'cont']. See explanation above
| pred_mode          | String| cont    | options: ['last', 'avg', 'cont']. See explanation above
| load_all           | Bool  | True    | This argument is now deprecated, always True
| attn_head          | Int   | 4       | Number of head in attn model
| lstm_layer         | Int   | 2       | Number of LSTM layer for lstm model (vanillalstm, bilstm, bilstmlm)
| lstm_dropout       | Float | 0.5     | Dropout rate of LSTM model
| lstm_hidden1       | Int   | 256     | Hidden unit in the first dense layer of lstm model
| lstm_hidden2       | Int   | 64      | Hidden unit in the second dense layer of lstm model
| eval               | Bool  | False   | Only evaluate model without training. Same as running inference.py with --part dev

Example:
```bash
python train.py --model mstc --batchsize 2 --epoch 20 --split 0 --lr 0.001 --lr_step_size 50 --lr_gamma 0.75n --train_mode active --load_all
```

### Inference

inference.py and inference-scene.py accept arguments:
- pretrained_model: Accept a list of pretrained_model found on the models/ folder, the naming of the models have to be ${model}_${accuracy}_dev
- load_all: (Deprecated) load all data into RAM. Now always True. If you don't have enough memory please download more RAM first.
- prob: accept 'small' or 'big', this is for aggregation where if majority voting failed and no of frames predicted to be the same, the probability should be taking the higher or lower. Always use big, obviously. This argument is a test of faith
- part: the default is 'test', but can use 'dev' to do inference on the validation data.

For inference-lm.py, the arguments are:
- pretrained_model: Accept a list of pretrained_model found on the models/ folder, the naming of the models have to be ${model}_${accuracy}_dev
- lm_path: path to the KenLM .arpa model
- part: the default is 'test', but can use 'dev' to do inference on the validation data.
- beam_size: beam size for beam search, default: 5
- threshold: threshold of class probability from the model to be considered as candidate in beam search, default: 0.2
- remove_zero: forcely ignore zero as candidate (except when there's no other candidate), default: False

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
3. Go to train.py append your model name into the arguments
```bash
parser.add_argument('--model', dest='model', default='mstcn',
                        choices=['vanillalstm', 'bilstm',
                                 'bilstmlm', 'attn', 'winattn',
                                 'bigru', 'attn', 'mstcn', 'ctcloss'], #TODO: add your model name here
```
4. Go to inference.py or inference-scene.py append in the else-if chain to initialize the network class
```bash
elif model == 'mstcn':
    net = MultiStageModel(400, n_class=n_class).to(device)
#TODO: add your model name here
# elif args.model == 'mymodel':
#   net = MyNet(<arguments>).to(device)
```

## Kaggle Result
Rank 13 with accuracy 0.74844