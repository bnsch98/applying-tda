from pathlib import Path
from pandas import read_parquet, concat, DataFrame
import sys

# create a joint dataset of embeddings from all datasets


def load_embeddings(datasets, numfiles=None):
    """
    Load embeddings from specified datasets and concatenate them into a single DataFrame.

    Parameters:
    - datasets: List of dataset names to load.
    - numfiles: Number of files to load from each dataset (if None, load all).

    Returns:
    - DataFrame containing concatenated embeddings.
    """
    embeddings_data = DataFrame()

    for dataset in datasets:
        # Determine the suffix based on the dataset
        suffix = "special" if dataset in ["aql", "aol"] else "all"

        # Path to embeddings
        path = Path(
            f"/home/benjamin/dev/applying-tda/data/{dataset}-get-embeddings-{suffix}")

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


if __name__ == "__main__":
    # Example usage
    datasets = ['aol', 'aql', 'ms-marco', 'orcas']
    # specify number of files to load from each dataset
    numfiles = int(sys.argv[1]) if len(
        sys.argv) > 1 else None  # Set to None to load all files
    embeddings_data = load_embeddings(datasets, numfiles)

    # write data to parquet file
    output_path = Path(
        f"/home/benjamin/dev/applying-tda/data/embeddings_combined_{numfiles*4}.parquet")
    embeddings_data.to_parquet(output_path, index=False)
    print(f"Combined embeddings saved to {output_path}")
