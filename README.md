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

## Key Components

### EncoderDecoder (`src/EncoderDecoder.py`)
This module provides functionality to:
- Encode citation network data into graph embeddings
- Combine embeddings from selected citation links to form a new paper representation
- Decode combined embeddings to generate coherent paper abstracts

### LinkPredictor (`src/LinkPredictor.py`)
This module implements algorithms to:
- Predict the optimal number of citation links for a new paper node
- Select specific papers to cite based on graph structural information
- Create meaningful citation relationships for the new node

### Testing and Evaluation (`src/testing_LinkPredictor.ipynb`)
This notebook provides comprehensive testing and validation of the link prediction algorithms:
- Evaluates the quality of link prediction on masked nodes using the Cora dataset as a benchmark
- Assesses how well the graph structure is preserved during the link prediction process
- Provides insights into the performance of different prediction strategies
- Validates that the new nodes integrate naturally into the existing citation network

## Setup and Usage

1. Clone this repository
2. Download the Citation Network Dataset from Kaggle
3. Run the pipeline in order:
   - Initial processing: `parseFile_raw.py`
   - Batch preprocessing: `parseFile_batched.py`
   - Data filtering: `get_filter_data_with_references.ipynb`
   - Analysis and model training: `MLNS_groupProject.ipynb`

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

## Contributors
- **Timothée Strouk**
- **Galdwin Leduc**
- **Luca Perna**
