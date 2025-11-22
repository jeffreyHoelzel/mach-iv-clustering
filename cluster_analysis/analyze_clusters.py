import argparse
from setup.preprocess import load_raw
from pipelineio.visualization import plot_mode_cluster_heatmaps, radar_chart
from setup.config import DATA_PATH, QUESTION_COLS
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def main() -> None:
    """Main script to run pipeline. Using k=2 as best seen in Jupyter Notebook testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--cluster-labels", type=str, required=True, help="Filepath to cluster labels dataframe csv")
    args = parser.parse_args()

    # read dfs
    X = pd.read_csv(args.cluster_labels, index_col=0)
    plot_mode_cluster_heatmaps(X, f"kmeans_response_heatmap")
    
    data = pd.read_csv(DATA_PATH)
    data = data.loc[X.index]
    other_columns = [col for col in data.columns.to_list() if not col.startswith("Q")]

    # add other columns to sampled df with clusters
    full_df = pd.concat([X, data[other_columns]], axis=1)
    
    other_answers = ["TIPI1","TIPI2","TIPI3","TIPI4","TIPI5","TIPI6","TIPI7","TIPI8","TIPI9","TIPI10"]
    other_answers_df = full_df[other_answers]

    other_answers_df = other_answers_df.rename(columns={
        "TIPI1":"Extraverted, enthusiastic",
        "TIPI2":"Critical, quarrelsome",
        "TIPI3":"Dependable, self-disciplined",
        "TIPI4":"Anxious, easily upset",
        "TIPI5":"Open to new experiences, complex",
        "TIPI6":"Reserved, quiet",
        "TIPI7":"Sympathetic, warm",
        "TIPI8":"Disorganized, careless",
        "TIPI9":"Calm, emotionally stable",
        "TIPI10":"Conventional, uncreative"
        })


    # scale other answers
    scaler = MinMaxScaler()
    normalized_df = pd.DataFrame(
        scaler.fit_transform(other_answers_df),
        columns=other_answers_df.columns,
        index=other_answers_df.index
    )

    # add clusters
    normalized_df = pd.concat([normalized_df, full_df["Cluster"]], axis=1)
    
    # graph each cluster
    cluster_dfs = normalized_df.groupby("Cluster")
    for cluster, df in cluster_dfs:
        df.drop("Cluster", axis=1, inplace=True)

        # get the modes
        modes = df.mode().iloc[0].to_list()

        radar_chart(df.columns.to_list(), modes, f"Cluster {cluster}")
    
    plot_mode_cluster_heatmaps(normalized_df, "other_responses_heatmap")


if __name__ == "__main__":
    main()