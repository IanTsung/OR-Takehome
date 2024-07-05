from utils.data_processing import DataProcessor
from utils.clustering import Clustering

def main():
    pdf_dir = "Green Energy Dataset"
    abstract_output = "output/abstracts.json"
    results_file = "output/output.csv"
    summary_file = "output/summary.csv"
    
    processor = DataProcessor(pdf_dir, abstract_output)
    preprocessed_abstracts, titles = processor.extract_abstracts(record_output=True)
    
    cluster = Clustering(preprocessed_abstracts, num_clusters=9)
    cluster.vectorise_documents()
    cluster.apply_pca(num_components=100)
    cluster.evaluate_clusters()
    cluster.perform_clustering()
    cluster.visualise_clusters(titles)
    cluster.save_results(titles, results_file, summary_file)

if __name__ == '__main__':
    main()