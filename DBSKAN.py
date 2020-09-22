import os

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


def find_classes(data):
    clustering = KMeans(n_clusters=60, random_state=0).fit(data)
    # clustering = DBSCAN(eps=0.5, min_samples=2).fit(data)
    return clustering.labels_


def main():
    data_dir = "data"
    file_path = os.path.join(data_dir, "anywhere_embeddings.npy")
    data = np.load(file_path, allow_pickle=True).tolist()
    make_emb_tsv(data, os.path.join(data_dir, "embeddings.tsv"))
    classes = find_classes(data)
    main_df = pd.read_json(os.path.join(data_dir, 'fixed_anywhere.json'), lines=True)
    make_meta_tsv(main_df, classes, os.path.join(data_dir, "meta.tsv"))
    make_classification_dataset(data, classes, os.path.join(data_dir, "classification_dataset.parquet"))


def make_emb_tsv(data, output_path):
    df = pd.DataFrame(data=data)
    df.to_csv(output_path, sep='\t', index=False, header=False)


def make_meta_tsv(df, dbskan_clusters, output_path):
    df = df.drop("count", axis=1)
    df.columns = ['id', 'skills']
    df['classes'] = pd.Series(dbskan_clusters)
    df.to_csv(output_path, sep='\t', index=False)


def make_classification_dataset(embeddings, classes, path):
    print(len(embeddings))
    df = pd.DataFrame(data={'embeddings': embeddings, 'classes': classes})
    df.to_parquet(path, engine='pyarrow')


if __name__ == "__main__":
    main()