import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import itertools

def visualize_pareto_front_3d(population_dict, save_path=None, dataset_name=""):
    """
    population_dict: dict[str, np.ndarray], 每个算法的名字对应一个 [N, 3] 的 Pareto 前沿数组
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 使用 matplotlib 自带颜色循环
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    for algo_name, pop in population_dict.items():
        color = next(color_cycle)
        makespan, load, idle = pop[:, 0], pop[:, 1], pop[:, 2]

        ax.scatter(makespan, load, idle, label=algo_name, color=color, alpha=0.6)
        # for x, y, z in zip(makespan, load, idle):
        #     ax.text(x, y, z, f'({x:.1f},{y:.1f},{z:.1f})', fontsize=8, color=color)

    ax.set_xlabel('Makespan')
    ax.set_ylabel('Load Balance')
    ax.set_zlabel('Idle Time')
    ax.set_title(f'3D Pareto Front - {dataset_name}')
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()
