import argparse
import tensorflow as tf
import numpy as np
from data_processing import parse_dataset_scanobjectnn, parse_dataset_modelnet
from model_training import augment, create_model, train_model, save_model, load_model
from visualization import visualize_points


def preprocess_dataset(dataset_folder, dataset, num_points):
    # Code to load and preprocess the dataset
    if(dataset == "scanobjectnn"):
	    all_points, all_labels, class_map = parse_dataset_scanobjectnn(dataset_folder, num_points)

	    # Convert the lists to NumPy arrays
	    all_points = np.array(all_points)
	    all_labels = np.array(all_labels)

	    # Get the number of samples
	    num_samples = len(all_points)

	    # Create an index array and shuffle it
	    indices = np.arange(num_samples)
	    np.random.shuffle(indices)

	    # Shuffle the arrays based on the shuffled indices
	    shuffled_points = all_points[indices]
	    shuffled_labels = all_labels[indices]

	    # Get the total number of samples
	    num_samples = len(shuffled_points)

	    # Calculate the split index
	    split_index = int(0.8 * num_samples)

	    # Split the data into training and validation sets
	    train_points = shuffled_points[:split_index]
	    train_labels = shuffled_labels[:split_index]
	    val_points = shuffled_points[split_index:]
	    val_labels = shuffled_labels[split_index:]

	    train_points = train_points.tolist()
	    train_labels = train_labels.tolist()
	    val_points = val_points.tolist()
	    val_labels = val_labels.tolist()

	    return train_points, train_labels, val_points, val_labels, class_map
    else:
    	 train_points, test_points, train_labels, test_labels, class_map = parse_dataset_scanobjectnn(dataset_folder, num_points)
    	 return train_points, train_labels, val_points, val_labels, class_map



def train_and_save_model(train_points, train_labels, val_points, val_labels, num_points, num_classes, epochs, batch_size,
                         model_save_path):
    train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((val_points, val_labels))

    train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(batch_size)
    test_dataset = test_dataset.shuffle(len(val_points)).map(augment).batch(batch_size)

    # Code to create and train the model
    model = create_model(num_points, num_classes)
    train_model(model, train_dataset, test_dataset, epochs)

    # Code to save the model
    save_model(model, model_save_path)


def load_and_visualize_model(model_path, test_dataset, num_samples, class_map):
    loaded_model = load_model(model_path)

    data = test_dataset.take(num_samples)

    points, labels = list(data)[0]
    points = points[:8, ...]
    labels = labels[:8, ...]

    preds = loaded_model.predict(points)
    preds = tf.math.argmax(preds, -1)

    points = points.numpy()

    # Code to visualize the points and predictions
    visualize_points(points, labels, preds, class_map)


def main(args):
    dataset_folder = args.dataset_folder
    num_points = args.num_points
    num_classes = args.num_classes
    epochs = args.epochs
    batch_size = args.batch_size
    model_save_path = args.model_save_path
    dataset = args.dataset
    visualize = args.visualize

    train_points, train_labels, val_points, val_labels, class_map = preprocess_dataset(dataset_folder, dataset, num_points)

    train_and_save_model(train_points, train_labels, val_points, val_labels, num_points, num_classes, epochs,
                         batch_size, model_save_path)

    test_dataset = tf.data.Dataset.from_tensor_slices((val_points, val_labels))

    if visualize:
        load_and_visualize_model(model_save_path, test_dataset, num_samples=6, class_map=class_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default="/content/object_dataset", help="Path to the dataset folder")
    parser.add_argument("--num_points", type=int, default=2048, help="Number of points")
    parser.add_argument("--num_classes", type=int, default=15, help="Number of classes")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--model_save_path", type=str, default="/content/weights", help="Path to save the model")
    parser.add_argument("--dataset", type=str, default="modelnet", help="Dataset you want to train. Options: modelnet and scanobjectnn")
    parser.add_argument("--visualize", type=bool, default=False, help="Whether to visualize the points")

    args = parser.parse_args()
    main(args)

