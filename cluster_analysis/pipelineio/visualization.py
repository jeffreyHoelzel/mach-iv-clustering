import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from .io_utils import save_fig
from setup.config import QUESTION_COLS
import textwrap

def plot_pca_clusters(X: pd.DataFrame,
                      filename: str, 
                      ks: tuple[int, ...] = (2, 3, 4)) -> None:
    """Creates a plot of the principal component analysis using provided cluster sizes."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(1, len(ks), figsize=(5 * len(ks), 5))
    for i, k in enumerate(ks):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(X)
        ax[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=15)
        ax[i].set_title(f"PCA, k={k}")
        ax[i].set_xlabel("PC1")
        ax[i].set_ylabel("PC2")
        
    plt.tight_layout()
    save_fig(fig, "plots", "pca", f"{filename}.png")

def plot_mode_cluster_heatmaps(df_labeled: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Creates a heatmap of the modes by cluster per question response and returns a DataFrame of the modes."""
    columns = df_labeled.columns.to_list()
    modes = df_labeled.groupby("Cluster")[[col for col in columns if col != "Cluster"]].agg(
        lambda x: x.mode().iloc[0]
    )

    fig, ax = plt.subplots(figsize=(15, 6))
    sns.heatmap(modes, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Mode MACH-IV Question Responses per Cluster")
    ax.set_xlabel("Questions")
    ax.set_ylabel("Clusters")

    save_fig(fig, "plots", "heatmaps", f"{filename}.png")
    return modes

def radar_chart(attributes, scores, title="Radar Chart"):
    """
    attributes: list of strings
    scores: list of floats (0–1)
    """

    N = len(attributes)

    # Close the polygon by appending the first value to the end
    values = scores + scores[:1]
    
    # Compute angles for each axis
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

    # Draw outline
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    # Add attribute labels
    ax.set_xticks(angles[:-1])

    wrapped = ["\n".join(textwrap.wrap(l, 12)) for l in attributes]
    ax.set_xticklabels(wrapped, fontsize=20)


    ax.set_title(title, y=1.1, fontsize=20)

    save_fig(fig, "plots", "radar", f"{title.replace(" ", "_")}.png")