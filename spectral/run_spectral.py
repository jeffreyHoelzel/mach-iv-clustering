from setup.preprocess import prep_sample
from clustering.cluster import label_and_score
from pipelineio.visualization import plot_mode_cluster_heatmaps, plot_spectral_embedding


def main() -> None:
    """
    Main script to run the spectral clustering pipeline.
    Uses k-NN + RBF spectral embedding and KMeans in embedding space.
    """
    # Load (and optionally sample) the data
    X = prep_sample(save=True, use_all=False)

    results, summary = label_and_score(
        X,
        ks=(2, 3, 4),
        save=True,
        prefix="spectral",
        n_neighbors=15,
    )
    print(summary)

    for k in (2, 3, 4):
        k_best = k
        labels_best = results[k_best]["labels"]
        embedding = results["embedding"]

        plot_spectral_embedding(embedding[:, :2], labels_best, f"spectral_embedding_k_{k}")
        
        df_labeled = X.copy()
        df_labeled["Cluster"] = labels_best

        plot_mode_cluster_heatmaps(df_labeled, f"spectral_response_heatmap_k_{k}")


if __name__ == "__main__":
    main()
