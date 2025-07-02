import numpy as np
import matplotlib.pyplot as plt

def generate_scatter():
    np.random.seed(0)
    points = np.random.rand(100, 3)
    np.save("../data/scatter_points.npy", points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    plt.title("Light Scatter Scene")
    plt.savefig("../proof/scatter_preview.png")
    plt.show()

if __name__ == "__main__":
    generate_scatter()
