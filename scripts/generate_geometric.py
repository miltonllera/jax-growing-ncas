# import matplotlib.pyplot as plt
from src.dataset.geometric import create_hash, generate_data, save_dataset


if __name__ == "__main__":
    class_list = ["cross", "square", "triangle", "circle"]
    sizes = ["medium"]  # Available sizes
    color = ["red", "blue", "green", "yellow", "white", "cyan", "magenta", ]
    rotation = [0.0]
    canvas_shape = 64, 64

    color_map = {
        "red": [1, 0, 0, 1],
        "blue": [0, 0, 1, 1],
        "green": [0, 1, 0, 1],
        "yellow": [1, 1, 0, 1],
        # "pink": [1, 0.75, 0.8, 1],
        "white": [1, 1, 1, 1],
        # "gray": [0.5, 0.5, 0.5, 1],
        # "brown": [0.5, 0.25, 0, 1],
        "cyan": [0, 1, 1, 1],
        # "purple": [0.5, 0, 0.5, 1],
        "magenta": [1, 0, 1, 1],
        # "indigo": [0.22, 0, 1, 1],
        # "gold": [1, 0.84, 0, 1],
        # "silver": [0.75, 0.75, 0.75, 1],
        # "turquoise": [0.25, 0.88, 0.82, 1],
        "black": [0, 0, 0, 1],
    }

    data, labels = generate_data(
        class_list,
        sizes,
        color,
        rotation,
        canvas_shape,
        color_map
    )

    hash = create_hash(len(data), (64, 64), class_list, color, 'one_hot', '')


    # plt.imshow(data[0])
    # plt.gca().set_facecolor('black')
    # plt.show()

    # plt.imshow(data[1])
    # plt.gca().set_facecolor('black')
    # plt.show()

    # plt.imshow(data[2])
    # plt.gca().set_facecolor('black')
    # plt.show()


    save_dataset(hash, data, labels, class_list, sizes, color, rotation)
