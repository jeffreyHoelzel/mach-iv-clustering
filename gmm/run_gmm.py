from setup.preprocess import prep_sample
from clustering.cluster import label_and_score
from pipelineio.visualization import plot_pca_clusters, plot_mode_cluster_heatmaps

def main() -> None:
    """Main script to run pipeline. Using k=2 as best seen in Jupyter Notebook testing."""
    X = prep_sample(save=True, use_all=False)
    results, summary = label_and_score(X, save=True)
    print(summary)

    k_best = 2
    labels_best = results[k_best]["labels"]

    df_labeled = X.copy()
    df_labeled["Cluster"] = labels_best

    plot_pca_clusters(X, "gmm_pca")
    cluster_modes = plot_mode_cluster_heatmaps(df_labeled, "gmm_response_heatmap")

if __name__ == "__main__":
    main()
