# Transfer Point-GNN, applied to MUT

This repository contains an extended version of
[Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud](http://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_Point-GNN_Graph_Neural_Network_for_3D_Object_Detection_in_a_CVPR_2020_paper.pdf), CVPR 2020.
applied to the MUT dataset by transferring a model that was pretrained on KITTI.

The code for the neural network was written in Python. 
A good part of it was taken from the code from Point-GNN and adapted appropriately 
to fulfill the requirements. Although the code structure was kept very similar to the original version, 
several parts had to be changed, restructured or expanded. 
For example it was necessary to expand the code such that it can handle more than 1 dataset, 
especially more than only the KITTI dataset. 
Also the possibility to freeze and unfreeze only certain layers had to be implemented 
in order to implement transfer learning and fine-tuning.

Several classes and functions were implemented to create and update the label files, 
folders and split for MUT and also to cut the very large point clouds from MUT in smaller pieces.
        
And finally a small framework for manual evaluation was implemented, too.

1. Prerequisites
2. Dataset, Data preprocessing and Structuring descriptions
3. Configuration   
4. How to use

## Prerequisites

As Point-GNN used Tensorflow 1.15, it is also used in this work, 
but updated to a tensorflow v2 compatible syntax.
The MUT cloud files are in .e57 format.

[Install CUDA](https://developer.nvidia.com/cuda-10.0-download-archive) if you want GPU support.

Install packages with conda and pip:
```
conda install cudatoolkit=11.0.221
conda install -c anaconda cudnn

pip install tensorflow-gpu==2.4.1
pip install --upgrade tf_slim
conda install -c conda-forge opencv 
conda install -c open3d-admin open3d
conda install -c conda-forge scikit-learn
conda install tqdm
conda install shapely
conda install matplotlib
```

for pye57 (reading of e57 data) follow https://github.com/davidcaron/pye57:
```
conda install xerces-c
pip install pye57
```
And change pye57 as https://github.com/davidcaron/pye57/issues/6#issuecomment-803894677 suggests

### Download Transfer Point-GNN
#### Transfer Point-GNN:
```
https://gitlab.lrz.de/ru38wej/transfer-learning-for-3d-object-detection-on-point-clouds.git
```

## 2. Dataset
###Structure

    DATASET_ROOT_DIR
    ├── tunnel                   # Tunnel point cloud files
    ├── labels                   # All labels (rot_sym_labels)
    └── 3DOP_splits              # Split files.
        ├── test.txt
        ├── train.txt
        └── val.txt

where the splits contain the label file names for the clouds that contain known and relevant objects.
Insert paths to dataset directorys into ```util.paths_and_data.py```

### Data preprocessing
For each cloud the label data are generated from the corresponding excel-table.
For code compability they are designed like the labels for the KITTI dataset.
See label files and ```dataset_classes.dataset.get_label``` for the exact label structure.
Consider also the differences in order of coordinates and measures (described in ```data_preprocessing.create_label_methods()```).
Also a rotation symmetric version of the labels is generated especially for the safety posts.

As the 100m cloud files (~60 000 000 pts) are too large for calculation they need to be splitted first.
A good size is 3 500 000 pts (roughly 5m), but it could also be further reduced to 1 750 000 pts. 
In order to reduce cutting objects, an overlap of 1 750 000 pts is introduced.
Also the labels and split files need to be splitted respectively.

Adapting as needed and running ```util.preprocess_data.py``` will do the job
(see comments in method for further details).

###Classes
14 classes are distinguished:
```
Anschlusskasten
Antenne
Beleuchtung
Fernsprecher
Geländer
Laufweg
Leiter
Magnet
Schild
Signal
Spiegeltür
SR-Stab(IST-Lage)
SR-Stab(SOLL-Lage)
Stütze
Uhr
```

###Label Methods
(each method includes a background and a DontCare class)
few_labels:
```
Schild
Signal
```
many_labels:
```
SR-Stab
Magnet
```

posts_rot: (without SR-Stab differences)
-> exists with different amount of posts: 128, 256, 512, 786, 1024
``` 
SR-Stab 
```

####Create new label method
	1. - Objekte auswählen
	2. - dict in data_specifics eintragen und in get_label_method ergänzen
	3. - split_file erstellen
		a. - label_file_operations.filter_labels_and_save_to_split()
	4. - Anzahl der im Split enthaltenen Objekte extrahieren
		a. - label_file_operations.get_MUT_split_object_count()
	5. - split file in train/val/test aufteilen
		a. - label_file_operations.split_labels()
	6. - Anzahl der im train split enthaltenen files extrahieren
		a. - # Zeilen im split (auch schon im obj count output enthalten)
	7. - config und meta_config in configs erstellen
		a. - configs anpassen
			i. - config
				1) - label_method
				2) - trainable (für jedes Layer)
				3) - num_classes
			ii. - meta_config
				1) - NUM_TEST_SAMPLE (=Anzahl Beispiele/Epoche; Anzahl der im Split enthaltenen files)
				2) - train_dataset (split_file)
				3) - val_dataset (which dataset(split-file) to validate on)
				4) - train_dir (checkpoint_dir: where to start learning from)
	8. - checkpoint erstellen und 
		a. - für finetuning entsprechende checkpoint-Daten aus KITTI adden
                - für from scratch learning nichts adden

See and adapt method ```data_preprocessing.create_label_methods()``` (takes care of steps 3-8)

##3. Configuration
Adapt config files if needed or for new label_methods.
###Config
Model and Data parameters (in original method: train_config)
```
downsample_by_voxel_size(initial downsampling when reading file; graph is built after application of this; none for full resolution)
graph_gen_kwargs
    base_voxel_size     (second layer graph is built after this is applied)
    level_configs       (for different levels of graph gen)
label_method            (see above)
model kwargs
    trainable           (set for each layer seperatly for finetuning specific layers)
num_classes
runtime_graph_gen_kwargs
    base_voxel_size     (downsampling for inference)
```
###Meta-Config
Hyperparameters and file settings (in original method: train_train_config)
```
decay_step      (steps after which the lr is reduced by decay_factor)
decay_factor    (factor by which the lr is reduced)
initial_lr
max_epoch
max_steps
NUM_GPU       The number of GPUs to use. We used two GPUs for the reference model. 
              If you want to use a single GPU, you might also need to reduce the batch size by half to save GPU memory.
              Similarly, you might want to increase the batch size if you want to utilize more GPUs. 
              Check the train.py for details.
batch size                  (higher batch size -> higher memory utilization)
load_dataset_to_mem         (true -> higher memory utilization)
num_load_dataset_workers    (multiprocessing)
train_dataset               (split file containing training file names)
train_dir                   (where to save to /start learning from)
```

##4. How to use
For transfer learning the source and target model need to have the same structure, i.e. same number of classes etc.
            
classes        Point-GNN                   MUT
1 (4)          car_auto_T3_train            posts_rot
2 (6)          ped_cycl_auto_T3_trainval     few_labels / many_labels
For each class there is a front-view and a side-view class,
and adittionally a Background and a DontCare class.


You can use tensorboard to view the training and evaluation status.
```
tensorboard --logdir=./train_dir
```

For training, also a ```statistics.json``` file is generated, 
which can be evaluated by using ```evaluation.train_statistics.evaluate_training(checkpoint_name)```.
This generates a plot and saves it under ```train_checkpoint_name.png```

##Pipeline
### Training
We put training parameters in a train_config file. To start training, we need both the meta_config and config.
```
config: model parameters
meta_config: system parameters (filepaths etc.)
```
```
usage: train.py [-h] [checkpoint CHECKPOINT_NAME]
                [dataset MUT/KITTI]
                [--dataset_root_dir DATASET_ROOT_DIR]
                [--dataset_split_file DATASET_SPLIT_FILE]
                [--label_dir LABEL_DIR]
                [--debug True/False]
                train_config_path config_path

Training of PointGNN

positional arguments:
  checkpoint     checkpoint name; code looks under ./configs/checkpoint_name_(meta_)config for configs
  dataset        MUT/KITTI

optional arguments:
  -h, --help            show this help message and exit
  --dataset_root_dir DATASET_ROOT_DIR
                        Path to KITTI dataset. Default="../dataset/kitti/"
  --dataset_split_file DATASET_SPLIT_FILE
                        Path to dataset split file.Default="DATASET_ROOT
                        _DIR/3DOP_splits/meta_config["train_dataset"]"
  --label_dir LABEL_DIR
                        Path to label directory. Exists for differentiation between rotation symmetric
                        and rotation asymmetric labels.
  --debug True/False
                        Enables print commands that help with debugging and shows analyzed point cloud cuts.
```
For example:
```
python3 train.py configs/MUT_train_config configs/MUT_system_config
```

### Inference
#### Run a checkpoint
Test on the validation split:
```
python3 run.py checkpoints/MUT --dataset_root_dir DATASET_ROOT_DIR --output_dir DIR_TO_SAVE_RESULTS
```
Test on the test dataset:
```
python3 run.py checkpoints/car_auto_T3_trainval/ --test --dataset_root_dir DATASET_ROOT_DIR --output_dir DIR_TO_SAVE_RESULTS
```

```
usage: run.py [-h] [dataset MUT/KITTI] [checkpoint_path CHECKPOINT_PATH] 
              [-l LEVEL] [--test] [--no-box-merge] [--no-box-score]
              [--dataset_root_dir DATASET_ROOT_DIR]
              [--dataset_split_file DATASET_SPLIT_FILE]
              [--output_dir OUTPUT_DIR]
              [--jobname JOBNAME]

Point-GNN inference on KITTI

positional arguments:
  dataset               MUT/KITTI
  checkpoint_path       Path to checkpoint

optional arguments:
  -h, --help            show this help message and exit
  -l LEVEL, --level LEVEL
                        Visualization level, 0 to disable,1 to nonblocking
                        visualization, 2 to block.Default=0
  --test                Enable test model
  --no-box-merge        Disable box merge.
  --no-box-score        Disable box score.
  --dataset_root_dir DATASET_ROOT_DIR
                        Path to KITTI dataset. Default="../dataset/kitti/"
  --dataset_split_file DATASET_SPLIT_FILE
                        Path to KITTI dataset split
                        file.Default="DATASET_ROOT_DIR/3DOP_splits/val.txt"
  --output_dir OUTPUT_DIR
                        Path to save the detection
                        results, Default="CHECKPOINT_PATH/eval/"
  --jobname JOBNAME
                        Saves predictions in a folder named with jobname.
```
### Evaluation
Manually evaluate by using ```evaluation.evaluate_predictions_manually.evaluate_manually()```

Automatically evaluate by using ```evaluation.eval_MUT.py```


Install kitti-object-eval-python evaluation:
https://github.com/traveller59/kitti-object-eval-python

Prerequisites:
```
conda install numpy
conda install scikit-image
conda install -c conda-forge fire
conda install -c numba cudatoolkit=10.0
```
Evaluate output results on the validation split:
```
python evaluate.py evaluate --label_path=/path/to/your_gt_label_folder --result_path=/path/to/your_result_folder --label_split_file=/path/to/val.txt --current_class=0 --coco=False
```
