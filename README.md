# Citation Network Analysis Project

## Overview
This project processes and analyzes the Citation Network Dataset from Kaggle to generate entirely new academic papers. The system creates new nodes in the citation graph by predicting link structures, combining embeddings, and generating abstracts based on the graph's structural information.

## Data Source
The original dataset comes from [Kaggle's Citation Network Dataset](https://www.kaggle.com/datasets/mathurinache/citation-network-dataset/data), which contains metadata about academic papers including their citations and references.

## Project Structure
```
.
├── data/
│   ├── dblp_papers_filtered.csv
│   └── dblp_papers_filtered_sample_with_refs.csv
├── preprocess_data_scripts/
│   ├── get_filter_data_with_references.ipynb
│   ├── parseFile_batched.py
│   └── parseFile_raw.py
├── src/
│   ├── CitationGraphAbstractGenerator.py
│   ├── EncoderDecoder.py
│   ├── LinkPredictor.py
│   └── testing_LinkPredictor.ipynb
├── MLNS_groupProject.ipynb
└── README.md
```

## Project Objective
The primary objective is to create entirely new paper nodes in the citation network:
1. Predict the number of citation links for a new paper node
2. Predict specific papers the new node should link to based on graph structural information
3. Combine embeddings from the selected links to create a representation of the new paper
4. Generate an abstract for this new paper based on the combined embeddings
5. Store the new node with its abstract and citation links in the graph

## Data Processing Pipeline

### 1. Initial Data Processing
The raw JSON data from Kaggle is processed and split into manageable batches using:
```
python3 parseFile_raw.py
```
This script divides the data into batches of 100,000 papers each, creating CSV files in the `data/dblp_batched` directory.

### 2. Batch Preprocessing
Each batch is then preprocessed using:
```
python3 parseFile_batched.py
```
This script performs several transformations:
- Formats reference IDs from semicolon-separated strings to lists of integers
- Parses indexed abstracts and reconstructs readable text
- Cleans and reorganizes the data structure
- Outputs processed files to `data/dblp_batched_processed`

### 3. Data Filtering
After preprocessing, relevant papers and their references are extracted using:
```
jupyter notebook preprocess_data_scripts/get_filter_data_with_references.ipynb
```
This notebook filters the dataset based on specific criteria and creates:
- `dblp_papers_filtered.csv`: The main filtered dataset
- `dblp_papers_filtered_sample_with_refs.csv`: A sample dataset containing 100 papers from 2020 along with all their cited papers, ideal for quick testing

### 4. Analysis and Model Training
The main analysis and modeling are conducted in the root-level notebook:
```
jupyter notebook MLNS_groupProject.ipynb
```
This notebook takes `data/dblp_papers_filtered_sample_with_refs.csv` as an input.
It reduces the dataset (adjust the values depending of your RAM constraints).
Then using the classes in `src/CitationGraphAbstractGenerator.py` and `src/LinkPredictor.py`
it will create a new node in the graph created from the dataset in input. In the process,
it will show the generated abstract. Finally, a visualization will be shown along with metrics.
At the bottom of the notebook the results of generation with different behavior and parameters can be observed.

## Key Components

### `/!\ Deprecated` EncoderDecoder (`src/EncoderDecoder.py`)
This module provides functionality to:
- Encode citation network data into graph embeddings
- Combine embeddings from selected citation links to form a new paper representation
- Decode combined embeddings to generate coherent paper abstracts

### LinkPredictor (`src/LinkPredictor.py`)
The EnhancedLinkPredictor class implements advanced algorithms to predict citation relationships for new paper nodes in the citation network. Key features include:

- Dynamic determination of the optimal number of citation links for a new paper node based on network statistics
- Selection of specific papers to cite using multiple structural metrics:
   - Personalized PageRank (weighted most heavily at 70%)
   - Jaccard Coefficient (15%)
   - Adamic-Adar Index (5%)
   - Resource Allocation Index (5%)
   - Centrality metrics (3%)
   - Preferential Attachment (2%)

- Probabilistic edge prediction with controlled randomness factor for exploration
- Community detection using Louvain algorithm for small graphs and degree-based approximation for large graphs
- Comprehensive network analysis capabilities to evaluate structural properties

The implementation includes sophisticated evaluation methods that:
- Assess how well graph structure is preserved during link prediction
- Compare network metrics between original, masked, and predicted networks
- Analyze changes in key properties like clustering coefficient, transitivity, assortativity, and path lengths
- Handle directed citation relationships appropriately using directed graph algorithms

This component is critical for maintaining the structural integrity of the citation network when predicting how new papers would integrate into the existing literature.RetryClaude can make mistakes. Please double-check responses.

### Testing and Evaluation (`src/testing_LinkPredictor.ipynb`)
This notebook provides comprehensive testing and validation of the link prediction algorithms:
- Evaluates the quality of link prediction on masked nodes using the Cora dataset as a benchmark
- Assesses how well the graph structure is preserved during the link prediction process
- Provides insights into the performance of different prediction strategies
- Validates that the new nodes integrate naturally into the existing citation network

### Add new nodes and generate associated abstract (`src/CitationGraphAbstractGenerator.py`)
The CitationGraphAbstractGenerator class contains the methods to:
- load a given citation graph
- compute the embedding of each node
- create new nodes (using LinkPredictor to have the nodes links) and its associated abstract using t5 small and gemma3:4b
- evaluate a node, using cosine similarity and, by extracting the abstract's concepts, concepts comparaison
- visualize the graph, emphazing the new nodes

This implementation aggregates embeddings doing an average of the first hop neighbors.
The aggregated embedding is decoded using t5 giving an abstract. This abstract is then enhanced using gemma3:4b.
For a given node the following metrics are computed to evaluate the abstract (similarities are computed with the node's neighbors):
- Average similarity
- Max similarity
- Min similarity
- Std similarity
- Average concept similarity
- New concepts count
- New concepts pourcentage

This class is used in the `MLNS_groupProject.ipynb` file to run the abstract generation and visualize the citation graph with new nodes.

## Setup and Usage

1. Clone this repository
2. Download the Citation Network Dataset from Kaggle
3. Run the pipeline in order:
   - Initial processing: `parseFile_raw.py`
   - Batch preprocessing: `parseFile_batched.py`
   - Data filtering: `get_filter_data_with_references.ipynb`
   - Analysis and generation of new abstracts: `MLNS_groupProject.ipynb`

### Testing the System
To test the paper generation system:
1. Use the main notebook at the root level:
   ```
   jupyter notebook MLNS_groupProject.ipynb
   ```
2. For quick testing with the sample dataset:
   - Use the provided sample file: `data/dblp_papers_filtered_sample_with_refs.csv`
   - Run the MLNS_groupProject.ipynb notebook with this sample data
   - The notebook contains all necessary code to test the full pipeline from link prediction to abstract generation

## Dependencies
- pandas
- torch
- scikit-learn
- matplotlib
- tqdm
- numpy
- glob
- re
- ast
- tqdm
- networkx
- transformers
- nltk
- ijson
- gc
- os
- ollama
- json

## Contributors
- **Timothée Strouk**
- **Galdwin Leduc**
- **Luca Perna**
