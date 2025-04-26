# Citation Network Analysis Project

## Overview
This project processes and analyzes the Citation Network Dataset from Kaggle to generate new paper abstracts using citation graph structures. The system leverages link prediction techniques to predict potential citations between papers and uses these predictions to generate new abstracts based on the graph embeddings.

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

## Project Objectives
- Predict potential citation links between academic papers using graph structural information
- Generate new paper abstracts based on the predicted citation network
- Create a system that can both predict the number of links and determine which specific links to form

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
- Generate new graph embeddings from predicted links
- Decode graph embeddings to generate paper abstracts

### LinkPredictor (`src/LinkPredictor.py`)
This module implements algorithms to:
- Predict potential links (citations) between papers based on graph structural information
- Determine both the number of links to predict and which specific links to choose

### Testing and Evaluation (`src/testing_LinkPredictor.ipynb`)
This notebook provides comprehensive testing and validation of the link prediction algorithms:
- Evaluates the quality of link prediction on masked nodes using the Cora dataset as a benchmark
- Assesses how well the graph structure is preserved during the link prediction process
- Provides insights into the performance of different prediction strategies
- Validates that the predicted citation network maintains essential structural properties of the original

## Setup and Usage

1. Clone this repository
2. Download the Citation Network Dataset from Kaggle
3. Run the pipeline in order:
   - Initial processing: `parseFile_raw.py`
   - Batch preprocessing: `parseFile_batched.py`
   - Data filtering: `get_filter_data_with_references.ipynb`
   - Analysis and model training: `MLNS_groupProject.ipynb`

### Quick Testing
For quick testing without processing the entire dataset:
1. Use the provided sample file: `data/dblp_papers_filtered_sample_with_refs.csv`
2. This sample contains 100 papers from 2020 along with all their cited papers
3. Run directly with the testing notebook or main project notebook:
   ```
   jupyter notebook src/testing_LinkPredictor.ipynb
   # or
   jupyter notebook MLNS_groupProject.ipynb
   ```

## Dependencies
- pandas
- numpy
- glob
- re
- ast
- tqdm
- Network/graph libraries (e.g., NetworkX)
- Deep learning frameworks for embedding generation and abstract prediction
- Other libraries as imported in the scripts

## Contributors
- **Timothée Strouk**
- **Galdwin Leduc**
- **Luca Perna**
