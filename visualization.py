import matplotlib.pyplot as plt

def visualize_points(points, labels, preds, class_map):
    fig = plt.figure(figsize=(15, 10))
    for i in range(8):
        ax = fig.add_subplot(2, 4, i+1, projection="3d")
        ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
        ax.set_title(
            "pred: {:}, label: {:}".format(
                class_map[preds[i].numpy()], class_map[labels.numpy()[i]]
            )
        )
        ax.set_axis_off()
    plt.show()

