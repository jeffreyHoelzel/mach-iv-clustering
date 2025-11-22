from setup.preprocess import prep_sample
from clustering.distances import compute_default_linkages
from clustering.cluster import label_and_score
from pipelineio.visualization import plot_dendrograms, plot_pca_clusters, plot_mode_cluster_heatmaps

def main() -> None:
    """Main script to run pipeline. Using Ward linkage as best linkage as seen in Jupyter Notebook testing."""
    X = prep_sample(save=True, use_all=True)
    Z_single, Z_complete, Z_average, Z_ward = compute_default_linkages(X)
    plot_dendrograms(Z_single, Z_complete, Z_average, Z_ward, "default_dendrograms")
    results, summary = label_and_score(X, Z_ward, save=True, linkage="ward")
    print(summary)

    plot_pca_clusters(X, Z_ward, f"ward_linkage_pca_k")

    for k in (2, 3, 4):
        k_best = k
        labels_best = results[k_best]["labels"]

        df_labeled = X.copy()
        df_labeled["Cluster"] = labels_best

        cluster_modes = plot_mode_cluster_heatmaps(df_labeled, f"ward_linkage_response_heatmap_k_{k}")

if __name__ == "__main__":
    main()
