from pathlib import Path
from pandas import read_parquet, concat, DataFrame
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from tdamapper.cover import CubicalCover
from tdamapper.learn import MapperAlgorithm
from tdamapper.core import mapper_connected_components
import numpy as np
import kmapper as km
import umap
import sklearn
import sklearn.manifold as manifold
import os
from time import time
import sys
import numpy as np


# This script loads embeddings from multiple datasets, applies Kepler Mapper for topological data analysis,
# and visualizes the results in an HTML file.


def load_embeddings(dataset, numfiles=None):
    """
    Load embeddings from specified datasets and concatenate them into a single DataFrame.

    Parameters:
    - datasets: Name of dataset to load. ['aol', 'aql', 'ms-marco', 'orcas', 'combined']
    - numfiles: Number of files to load from each dataset (if None, load all).

    Returns:
    - DataFrame containing embeddings.
    """

    embeddings_data = DataFrame()
    # Determine the suffix based on the dataset
    suffix = "special" if dataset in ["aql", "aol"] else "all"

    # Path to embeddings
    path = Path(
        f"/home/benjamin/dev/applying-tda/data/{dataset}-get-embeddings-{suffix}") if dataset != "joint" else Path(f"/home/benjamin/dev/applying-tda/data/embeddings_combined_{numfiles*4}.parquet")

    if dataset == "joint":
        # Load combined embeddings directly from parquet file
        embeddings_data = read_parquet(path)
    else:
        # Get number of files in path
        files = list(path.glob("*.parquet"))
        print(f"Number of files ({dataset}): {len(files)}")

        # Limit to specified number of files if numfiles is set
        if numfiles is not None:
            files = files[:numfiles]

        # Load embeddings from each file
        for cnt, file_path in enumerate(files):
            print(f"Loading {cnt+1}/{len(files)} {dataset} {file_path.name}")
            df = read_parquet(file_path)
            # concatenate column with an identifier for the dataset
            df['dataset'] = dataset.upper()
            embeddings_data = concat([embeddings_data, df], ignore_index=True)

    return embeddings_data


def process_dataset(dataset: DataFrame, numfiles: int, dataname: str, visualize: bool = False):
    """
    Process a single dataset by applying Kepler Mapper, and visualizing the results.

    Parameters:
    - dataset: pandas dataframe.
    """
    start = time()

    print(dataset.shape)
    print(dataset.columns)

    # change dtype of arrays in the embeddings column to float32
    dataset["embeddings"] = dataset["embeddings"].apply(
        lambda x: np.array(x, dtype=np.float32))
    # convert to numpy array, standardize data
    embeddings = dataset.to_numpy()

    # Stack the arrays in the embeddings column into a 2D array
    emb_array = np.stack(embeddings[:, 1])

    # Standardize each feature (column-wise)
    emb_array = (emb_array - np.mean(emb_array, axis=0)) / \
        np.std(emb_array, axis=0)

    print(emb_array.shape)

    # initialize Kepler Mapper
    mapper = km.KeplerMapper(verbose=1)

    # project data into 2D subsapce via 2 step transformation, 1)isomap 2)UMAP
    projected_data = mapper.fit_transform(emb_array, projection=[manifold.Isomap(
        n_components=100, n_jobs=-1), umap.UMAP(n_components=2, random_state=1)])

    # cluster data using DBSCAN
    G = mapper.map(projected_data, emb_array,
                   clusterer=sklearn.cluster.DBSCAN(metric="cosine"))

    # define an excessively long filename (helpful if saving multiple Mapper variants for single dataset)
    fileID = 'projection=' + G['meta_data']['projection'].split('(')[0] + '_' + \
        'n_cubes=' + str(G['meta_data']['n_cubes']) + '_' + \
        'perc_overlap=' + str(G['meta_data']['perc_overlap']) + '_' + \
        'clusterer=' + G['meta_data']['clusterer'].split('(')[0] + '_' + \
        'scaler=' + G['meta_data']['scaler'].split('(')[0]

    # visualize graph
    if visualize:
        if dataname == "joint":
            origin = embeddings[:, 2]
            # set color values: for aql set 1, for remaining datasets set 0
            color_values = np.where(origin == 'AQL', 1, 0)
            # visualize the graph, for each node the majority of the origins should encode the color
            mapper.visualize(graph=G,
                             path_html=f"../data/mapper_{dataname}_NumFiles_{str(numfiles)}_{fileID}.html",
                             title=fileID,
                             custom_tooltips=embeddings[:, 0],
                             #  colorscale='binary',
                             color_values=color_values,
                             color_function_name='binary dataset origin',
                             node_color_function=np.array(
                                 ['average', 'std', 'sum', 'max', 'min']),
                             )
        else:
            # visualize the graph, for each node the majority of the origins should encode the color
            mapper.visualize(graph=G,
                             path_html=f"../data/mapper_{dataname}_NumFiles_{str(numfiles)}_{fileID}.html",
                             title=fileID,
                             custom_tooltips=embeddings[:, 0],
                             color_function_name='Log Percent Returns',
                             node_color_function=np.array(
                                 ['average', 'std', 'sum', 'max', 'min']),
                             )
        # move html file to data folder
        os.system(
            f"mv ../data/mapper_{dataname}_NumFiles_{str(numfiles)}_{fileID}.html /mnt/c/Users/Benjamin/Desktop/mapper-results/")

    end = time()
    print(f"Time taken for {dataname}: {(end - start)/60} minutes")
    return G


def kmapper_pipeline(mode: str = "seperate", dataset: str = "aol", numfiles: int = 45, visualize: bool = False):
    """
    Run the TDA pipeline for specified datasets.

    Parameters:
    - mode: 'seperate' to run for each dataset separately, 'joint' to run for all datasets combined
    - dataset: Name of the dataset to process (used in 'seperate' mode).
    - numfiles: Number of files to load from each dataset (if None, load all).
    """
    assert mode in ["seperate",
                    "joint"], "Mode must be either 'seperate' or 'joint'."
    assert mode == "joint" and dataset is None, "dataset must be None in 'joint' mode."

    if mode == "seperate" and dataset is None:
        datasets = ['aol', 'aql', 'ms-marco', 'orcas']
        for dataset in datasets:
            # loading data
            data = load_embeddings(dataset, numfiles)
            mapping = process_dataset(data, numfiles, dataname=dataset,
                                      visualize=visualize)
    elif mode == "seperate" and dataset is not None:
        # loading data for a single dataset
        data = load_embeddings(dataset, numfiles)
        mapping = process_dataset(
            data, numfiles, dataname=dataset, visualize=visualize)
    elif mode == "joint":
        # loading data for all datasets combined
        data = load_embeddings("joint", numfiles)
        # Process the dataset(s)
        mapping = process_dataset(
            data, numfiles, dataname='joint', visualize=visualize)
    else:
        raise ValueError("Invalid mode. Choose 'seperate' or 'joint'.")
    return mapping


def dict_print(d: dict, output_path: Path) -> None:
    stack = d.items()
    while stack:
        k, v = stack.pop()
        if isinstance(v, dict):
            stack.extend(v.iteritems())
        else:
            # print("%s: %s" % (k, v))
            with open(output_path, 'a') as f:
                f.write("%s: %s" % (k, v))


if __name__ == "__main__":
    # Usage through command line arguments or directly call the function
    mode = sys.argv[1] if len(sys.argv) > 1 else "joint"
    numfiles = int(sys.argv[2]) if len(sys.argv) > 2 else None
    visualize = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False
    dataset = sys.argv[4] if len(sys.argv) > 4 else None

    mapping = kmapper_pipeline(mode=mode, dataset=dataset, numfiles=int(
        numfiles) if numfiles is not None else None, visualize=visualize)

    print(mapping)
    print(type(mapping))
    # write mapping to file
    output_path = Path(
        f"/home/benjamin/dev/applying-tda/data/mapping_{mode}_NumFiles_{str(numfiles)}.txt")
    dict_print(mapping, output_path)
