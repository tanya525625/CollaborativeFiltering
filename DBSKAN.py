import os

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


def find_classes(data):
    clustering = KMeans(n_clusters=300, random_state=0).fit(data)
    # clustering = DBSCAN(eps=0.5, min_samples=2).fit(data)
    return clustering.labels_


def main():
    data_dir = "data"
    file_path = os.path.join(data_dir, "DBSKAN_embeddings.npy")
    data = np.load(file_path, allow_pickle=True).tolist()
    make_emb_tsv(data, os.path.join(data_dir, "embeddings.tsv"))
    classes = find_classes(data)
    main_df = pd.read_csv(os.path.join(data_dir, 'filtered_dataset.csv'))
    make_meta_tsv(main_df, classes, os.path.join(data_dir, "meta.tsv"))
    make_classification_dataset(data, classes, os.path.join(data_dir, "classification_dataset.parquet"))


def make_emb_tsv(data, output_path):
    df = pd.DataFrame(data=data)
    df.to_csv(output_path, sep='\t', index=False, header=False)


def make_meta_tsv(df, dbskan_clusters, output_path):
    df = df.drop('Unnamed: 0', axis=1)
    df.columns = ['id', 'skills', 'position']
    df['classes'] = pd.Series(dbskan_clusters)
    df.to_csv(output_path, sep='\t', index=False)


def make_classification_dataset(embeddings, classes, path):
    df = pd.DataFrame(data={'embeddings': embeddings, 'classes': classes})
    df.to_parquet(path, engine='pyarrow')


if __name__ == "__main__":
    main()