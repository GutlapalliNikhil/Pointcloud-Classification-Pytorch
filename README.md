# Point Cloud Classification using PyTorch

This repository contains code for training and visualizing a point cloud classification model using PyTorch. The model is based on the PointNet architecture and can be trained on two different datasets: ModelNet and ScanObjectNN.

## Prerequisites:

Make sure you have the following dependencies installed:

Python 3\
TensorFlow\
NumPy\
Matplotlib

## Dataset:

The repository provides support for two datasets: ModelNet10 and ScanObjectNN.

#### ModelNet10:

To train the model on the ModelNet10 dataset, you can download it by running the following command:

```console
wget http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
```

Then, extract the downloaded zip file by running the following command:

```console
unzip ModelNet10.zip > /dev/null
```

#### ScanObjectNN:

To train the model on the ScanObjectNN dataset, you can download it by running the following command:

```console
wget https://hkust-vgd.ust.hk/scanobjectnn/raw/object_dataset.zip
```

Then, extract the downloaded zip file by running the following command:

```console
unzip object_dataset.zip > /dev/null
```

## Usage:

You can use the **'main.py'** script to train and visualize the point cloud classification model. The available command-line arguments are as follows:

* **'--dataset_folder'** (default: "/content/object_dataset"): Path to the dataset folder.
* **'--num_points'** (default: 2048): Number of points in each point cloud.
* **'--num_classes'** (default: 15): Number of classes in the dataset.
* **'--epochs'** (default: 25): Number of training epochs.
* **'--batch_size'** (default: 32): Batch size for training.
* **'--model_save_path'** (default: "/content/weights"): Path to save the trained model.
* **'--dataset'** (default: "modelnet"): Dataset you want to train. Options: "modelnet" and "scanobjectnn".
* **'--visualize'** (default: False): Whether to visualize the points after training.

To train the model on the ModelNet dataset, run the following command:

```console
python3 main.py --dataset_folder <path_to_modelnet_dataset> --dataset modelnet --visualize
```

To train the model on the ScanObjectNN dataset, run the following command:

```console
python3 main.py --dataset_folder <path_to_modelnet_dataset> --dataset scanobjectnn --visualize
```

Note: Replace **'<path_to_modelnet_dataset>'** and **'<path_to_scanobjectnn_dataset>'** with the actual paths to the ModelNet10 and ScanObjectNN datasets respectively.

## Additional Files

* **'data_processing.py'**: Contains functions for loading and preprocessing the dataset.
* **'model_training.py'**: Contains functions for creating, training, and saving the point cloud classification model.
* **'visualization.py'**: Contains functions for visualizing the points and predictions.

## Results

After training the model, the script will save the trained model to the specified **'model_save_path'**. If the **'visualize'** flag is set to **'True'**, the script will also display visualizations of the points and predictions.

## Acknowledgments

The code in this repository is based on the PointNet architecture proposed in the following paper:

* Charles R. Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas. "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

## References

* ModelNet: https://modelnet.cs.princeton.edu/
* ScanObjectNN: https://hkust-vgd.ust.hk/scanobjectnn/

For more details on the implementation and usage, please refer to the source code files and the documentation within the files.
