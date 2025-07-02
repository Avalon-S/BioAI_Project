import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_solution_diversity(all_solutions, save_path=None):
    solution_vectors = []
    for solution in all_solutions:
        flat = []
        for job_schedule in solution:
            flat.extend(job_schedule)
        solution_vectors.append(flat)
    solution_vectors = np.array(solution_vectors)

    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    embeddings = tsne.fit_transform(solution_vectors)

    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], s=30, c='purple')
    plt.title("Solution Structural Diversity via t-SNE")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()
