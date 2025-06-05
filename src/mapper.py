# load data
from pathlib import Path
from pandas import read_parquet, concat, DataFrame
import numpy as np
import kmapper as km
from kmapper.jupyter import display
import umap
import sklearn
import sklearn.manifold as manifold
import os

for dataset in ['aol', 'aql', 'ms-marco', 'orcas']:
    # select dataset
    if dataset in ["aql", "aol"]:
        suffix = "special"
    else:
        suffix = "all"

    # path to embeddings
    path = Path(
        f"/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/analysis/{dataset}-get-embeddings-{suffix}")

    # get number of files in path
    files = len(list(path.glob("*.parquet")))
    print(f"Number of files: {files}")

    # set number of files to load
    # numfiles = 5
    numfiles = None
    # load embeddings
    embeddings_data = DataFrame()
    for cnt, path in enumerate(path.glob("*.parquet")):
        print(f"Loading {cnt+1}/{files} {path.name}")
        df = read_parquet(path)
        embeddings_data = concat([embeddings_data, df], ignore_index=True)
        # limit to files for testing
        if cnt+1 == numfiles:
            break
    print(embeddings_data.shape)
    print(embeddings_data.columns)

    # change dtype of arrays in the embeddings column to float32
    embeddings_data["embeddings"] = embeddings_data["embeddings"].apply(
        lambda x: np.array(x, dtype=np.float32))
    # convert to numpy array, standardize data
    embeddings = embeddings_data.to_numpy()

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
    mapper.visualize(G,
                     path_html=f"../data/mapper_{dataset}_NumFiles_{str(files)}_{fileID}.html",
                     title=fileID,
                     custom_tooltips=embeddings_data.iloc[:, 0].to_numpy(),
                     color_function_name='Log Percent Returns',
                     node_color_function=np.array(['average', 'std', 'sum', 'max', 'min']))

    # move html file to data folder
    os.system(
        f"mv ../data/mapper_{dataset}_NumFiles_{str(files)}_{fileID}.html /mnt/c/Users/Benjamin/Desktop/mapper-results/")
