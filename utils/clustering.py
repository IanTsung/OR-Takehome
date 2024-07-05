import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

class Clustering:
  def __init__(self, documents, num_clusters) -> None:
    self.documents = documents
    self.num_clusters = num_clusters
    self.vectoriser = TfidfVectorizer(stop_words="english", max_df=0.7, min_df=0.1)
    self.model = None
    self.vectors = None
    self.cluster_labels = None
    
  def vectorise_documents(self):
    self.vectors = self.vectoriser.fit_transform(self.documents)
    
  def apply_pca(self, num_components=100):
    max_components = min(self.vectors.shape[0], self.vectors.shape[1])
    if num_components > max_components:
      num_components = max_components - 1
    
    pca = PCA(n_components=num_components, random_state=42)
    self.vectors = pca.fit_transform(self.vectors.toarray())
    
  def evaluate_clusters(self, max_clusters=15):
    silhouette_scores = []
    
    for i in range(2, max_clusters):
      kmeans = KMeans(n_clusters=i, random_state=42)
      kmeans.fit(self.vectors)
      silhouette_scores.append(silhouette_score(self.vectors, kmeans.labels_))
      
    plt.figure(figsize=(8, 8))
    plt.plot(range(2, max_clusters), silhouette_scores, marker='o')
    plt.title("Silhouette")
    plt.xlabel("cluster nums")
    plt.ylabel("silhouette score")
    
    plt.tight_layout()
    plt.show()
    
  def perform_clustering(self):
    self.model = KMeans(n_clusters=self.num_clusters, random_state=42, init='k-means++', n_init=10, max_iter=300)
    self.model.fit(self.vectors)
    self.cluster_labels = self.model.labels_
    
  def visualise_clusters(self, titles):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='random')
    results = tsne.fit_transform(self.vectors)
    
    df = pd.DataFrame({
      "Title": titles,
      "Document": self.documents,
      "Cluster": self.cluster_labels,
      "TSNE1": results[:, 0],
      "TSNE2": results[:, 1]
    })
    
    plt.figure(figsize=(16, 12))
    sns.scatterplot(x='TSNE1', y='TSNE2', hue='Cluster', data=df, palette='tab10')
    
    for i in range(df.shape[0]):
      plt.text(df.TSNE1[i], df.TSNE2[i], df.Title[i], fontsize=5)
      
    plt.title("Clustering Visualisation")
    plt.xlabel("TSNE1")
    plt.ylabel("TSNE2")
    plt.legend(title="Cluster")
    plt.show()
    
  def get_key_terms(self, cluster):
    centroids = self.model.cluster_centers_[cluster]
    sorted_centroids = centroids.argsort()[::-1]
    
    terms = self.vectoriser.get_feature_names_out()
    top_terms = nltk.pos_tag([terms[i] for i in sorted_centroids[:50]])
    terms = [term for term, pos in top_terms if pos in ('NN', 'NNS', 'NNP', 'NNPS')]
    
    np.random.shuffle(terms)
    return ', '.join(terms[:7])
    
  def save_results(self, titles, results_file, summary_file):
    results_df = pd.DataFrame({
      "Title": titles,
      "Cluster": self.cluster_labels
    })
    
    results_df.to_csv(results_file, index=False)
    
    summary_df = pd.DataFrame({
      "Document": self.documents,
      "Cluster": self.cluster_labels
    })
    
    summary = summary_df.groupby("Cluster")["Document"].count().reset_index()
    summary.columns = ["Cluster", "Num of Papers"]
    summary["Key Terms"] = summary["Cluster"].apply(self.get_key_terms)
    
    summary.to_csv(summary_file, index=False)