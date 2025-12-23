from pathlib import Path
from pandas import read_parquet, concat, DataFrame
import numpy as np
import os
from time import time
import sys
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# This script loads embeddings from multiple datasets, applies Kepler Mapper for topological data analysis,
# and visualizes the results in an HTML file.


def load_embeddings(dataset, numfiles=None) -> DataFrame:
    """
    Load embeddings from specified datasets and concatenate them into a single DataFrame.

    Parameters:
    - datasets: Name of dataset to load. ['aol', 'aql', 'ms-marco', 'orcas', 'combined']
    - numfiles: Number of files to load from each dataset (if None, load all).

    Returns:
    - DataFrame containing embeddings.
    """
    assert dataset in ["aol", "aql", "ms-marco",
                       "orcas", "joint"], "Invalid dataset specified"
    assert numfiles is None or isinstance(
        numfiles, int), "numfiles must be an integer or None"
    # exclude 'joint' and numfiles == None
    if dataset == "joint" and numfiles is None:
        raise ValueError(
            "numfiles must not be None when loading joint dataset. For joint dataset, numfiles can be 3, 10, 15 or 20.")

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


def get_persistence_diagram(data: DataFrame, numfiles: int, dataname: str) -> np.ndarray:
    """
    Process a single dataset by applying Kepler Mapper, and visualizing the results.

    Parameters:
    - data: DataFrame containing embeddings.
    - numfiles: Number of files processed.
    - dataname: Name of the dataset.
    """
    # Placeholder for persistence diagram processing
    print(
        f"Processing {dataname} with {numfiles} files...get persistence diagram")

    # get vietoris rips persistence transformer
    # Parameter explained in the text
    VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])

    # Fit and transform the data to get persistence diagrams
    # Remove dataset identifier for processing
    data = data.drop(columns=['dataset'])
    # Drop query column
    data = data.drop(columns=['serp_query_text_url'])
    # Convert DataFrame to numpy array for processing

    data = np.vstack(data['embeddings'])
    # Ensure data is in the correct format (float32)
    data = data.astype(np.float32)
    # Print the shape of the data
    print(data.shape)

    # Fit the Vietoris-Rips persistence transformer to the data
    persistence_diagrams = VR.fit_transform(data)
    print(f"persistance diagram shape: {persistence_diagrams.shape}")
    return persistence_diagrams


def get_topological_features(persistence_diagrams: np.ndarray) -> np.ndarray:
    """
    Calculate persistence entropy from persistence diagrams.

    Parameters:
    - persistence_diagrams: Numpy array of persistence diagrams.

    Returns:
    - Numpy array of persistence entropy values.
    """
    # get persistence entropy transformer
    PE = PersistenceEntropy()

    # Fit and transform the persistence diagrams to get entropy values
    persistence_entropy = PE.fit_transform(persistence_diagrams)

    print(f"persistence entropy shape: {persistence_entropy.shape}")
    return persistence_entropy


def train_classifier(features: np.ndarray, labels: np.ndarray) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on the topological features.

    Parameters:
    - features: Numpy array of topological features.
    - labels: Numpy array of labels corresponding to the features.

    Returns:
    - Trained Random Forest classifier.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42)

    # Create a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the classifier on the training data
    clf.fit(X_train, y_train)

    # Print the accuracy on the test set
    print(f"Classifier accuracy: {clf.score(X_test, y_test)}")

    return clf


if __name__ == "__main__":

    # specify number of files to load from each dataset
    numfiles = int(sys.argv[1]) if len(
        sys.argv) > 1 else None  # Set to None to load all files

    # Example usage
    datasets = sys.argv[2:] if len(sys.argv) > 2 else ['aol', 'aql']

    for dataset in datasets:
        embeddings_data = load_embeddings(dataset, numfiles)
        print(embeddings_data.columns)
        # Process the dataset to get persistence diagrams
        persistence_diagrams = get_persistence_diagram(
            embeddings_data, numfiles, dataset)

        # Get topological features from the persistence diagrams
        topological_features = get_topological_features(persistence_diagrams)

        # Assuming labels are available in the embeddings_data DataFrame
        # Use dataset identifier as labels
        labels = embeddings_data['dataset'].values

        # Train a classifier on the topological features
        classifier = train_classifier(topological_features, labels)

        print(f"Finished processing {dataset}")
