from setup.preprocess import prep_sample
from clustering.cluster import label_and_score
from pipelineio.visualization import plot_pca_clusters, plot_mode_cluster_heatmaps

def main() -> None:
    """Main script to run pipeline. Using k=2 as best seen in Jupyter Notebook testing."""
    X = prep_sample(save=True, use_all=True)
    results, summary = label_and_score(X, save=True)
    print(summary)

    plot_pca_clusters(X, "kmeans_pca")

    for k in (2, 3, 4): 
        k_best = k
        labels_best = results[k_best]["labels"]

        df_labeled = X.copy()
        df_labeled["Cluster"] = labels_best

        cluster_modes = plot_mode_cluster_heatmaps(df_labeled, f"kmeans_response_heatmap_k_{k}")

if __name__ == "__main__":
    main()
