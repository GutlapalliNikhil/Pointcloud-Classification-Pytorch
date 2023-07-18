import os
import glob
import struct
import open3d as o3d
import numpy as np

def parse_dataset_scanobjectnn(dataset_folder, num_points=2048):

    # Iterate through the folders and remove unnessary files
    for folder_name in os.listdir(dataset_folder):
        folder_path = os.path.join(dataset_folder, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('_indices.bin'):
                    file_path = os.path.join(folder_path, file_name)
                    os.remove(file_path)
                if file_name.endswith('_part.bin'):
                    file_path = os.path.join(folder_path, file_name)
                    os.remove(file_path)
                if file_name.endswith('_part.xml'):
                    file_path = os.path.join(folder_path, file_name)
                    os.remove(file_path)
                    
    all_points = []
    all_labels = []
    class_map = {}
    folders = sorted(glob.glob(os.path.join(dataset_folder, "*")))

    for i, folder in enumerate(folders):
        class_map[i] = folder.split("/")[-1]
        total_files = glob.glob(os.path.join(folder, "*"))

        for file_name in total_files:
            try:
                bin_file = file_name
                point_format = '11f'
                with open(bin_file, 'rb') as file:
                    total_points = struct.unpack('f', file.read(4))[0]
                    point_cloud = o3d.geometry.PointCloud()
                    for _ in range(int(total_points)):
                        point_data = struct.unpack(point_format, file.read(44))
                        x, y, z, _, _, _, _, _, _, _, _ = point_data
                        point_cloud.points.append([x, y, z])

                points_np = np.asarray(point_cloud.points)

                if points_np.shape[0] > num_samples:
                    indices = np.random.choice(points_np.shape[0], num_samples, replace=False)
                    sampled_points = points_np[indices]
                else:
                    sampled_points = points_np

                all_points.append(sampled_points)
                all_labels.append(i)
            except:
                pass

    return np.array(all_points), np.array(all_labels), class_map


def parse_dataset_modelnet(dataset_folder, num_points = 2048):
  train_points = []
  train_labels = []
  test_points = []
  test_labels = []
  class_map = {}
  folders = glob.glob(os.path.join(dataset_folder, "[!README]*"))

  for i, folder in enumerate(folders):
    print("Processing folder: ", folder)

    # Store folder name with ID, so that we can retrieve later
    class_map[i] = folder.split("/")[-1]

    # gather all files
    train_files = glob.glob(os.path.join(folder, "train/*"))
    test_files = glob.glob(os.path.join(folder, "test/*"))

    for f in train_files:
      train_points.append(trimesh.load(f).sample(num_points))
      train_labels.append(i)


    for f in test_files:
      test_points.append(trimesh.load(f).sample(num_points))
      test_labels.append(i)

  return(
      np.array(train_points),
      np.array(test_points),
      np.array(train_labels),
      np.array(test_labels),
      class_map
  )
