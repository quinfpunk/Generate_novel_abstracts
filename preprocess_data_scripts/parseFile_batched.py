import os
import pandas as pd
import glob
import re
import ast
from tqdm import tqdm
import numpy as np

# Define the transformation functions
def referenceformating(cites):
    if not isinstance(cites, str):
        return []
    elif len(cites) == 0:
        return []
    try:
        valid_cites = [int(cid) for cid in cites.split(';')]
        return valid_cites
    except ValueError:
        return []

def parse_indexed_abstract(abstract_str):
    """Safely parse the indexed abstract which is stored as a Python dict literal"""
    if not isinstance(abstract_str, str) or not abstract_str.strip():
        return {}
    
    try:
        # Use ast.literal_eval to safely evaluate the Python dictionary string
        return ast.literal_eval(abstract_str)
    except (SyntaxError, ValueError):
        return {}

def reconstruct_abstract(indexed_abstract):
    """Convert the indexed abstract into readable text"""
    # If not a dictionary or None/empty, return empty string
    if not isinstance(indexed_abstract, dict) or not indexed_abstract:
        return ""
    
    # Check if it has the expected keys
    if "IndexLength" not in indexed_abstract or "InvertedIndex" not in indexed_abstract:
        return ""
    
    try:
        index_length = indexed_abstract["IndexLength"]
        inverted_index = indexed_abstract["InvertedIndex"]
        
        # Create a list of the correct length, initialized with empty strings
        abstract_words = [""] * index_length
        
        for word, positions in inverted_index.items():
            for pos in positions:
                if 0 <= pos < index_length and word.lower()!='abstract':  # Prevent index out of range errors
                    abstract_words[pos] = word
        
        # Join the words with spaces to form the abstract
        abstract = " ".join(abstract_words)
        return abstract
    except Exception:
        return ""
    
# Extract batch number and sort files numerically
def extract_batch_number(filename):
    match = re.search(r'papers_batch_(\d+)\.csv', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return 0

def preprocess_batch_files(output_directory):
    for batch_file in tqdm(batch_files, desc="Processing batch files"):
            batch_number = extract_batch_number(batch_file)
            
            try:
                # Read the CSV file - only read necessary columns to save memory
                data = pd.read_csv(
                    batch_file, 
                    sep=';',
                    usecols=lambda x: x in ['id', 'title', 'year', 'references', 'indexedAbstract']
                )
                
                # Apply the reference formatting
                data['references'] = data['references'].apply(referenceformating)
                
                # Process indexed abstracts if they exist
                if 'indexedAbstract' in data.columns:
                    # Parse the indexed abstracts using ast.literal_eval
                    data['indexedAbstract'] = data['indexedAbstract'].apply(parse_indexed_abstract)
                    
                    # Reconstruct the abstracts
                    data['abstract'] = data['indexedAbstract'].apply(reconstruct_abstract)
                data.drop(columns=['indexedAbstract'], inplace=True, errors='ignore')
                # Save the transformed data
                output_file = os.path.join(output_directory, f"processed_{os.path.basename(batch_file)}")
                data.to_csv(output_file, index=False, sep=';')
                # Clean up to free memory
                del data

            except Exception as e:
                tqdm.write(f"Error processing batch #{batch_number}: {e}")

if __name__ == "__main__":
    # Directory containing the batch files
    files_directory = "../data/dblp_batched"
    output_directory = "../data/dblp_batched_processed"
    
    # Find all batch files
    batch_files = glob.glob(os.path.join(files_directory, "papers_batch_*.csv"))
    
    # Sort the batch files by their numerical batch number
    batch_files.sort(key=extract_batch_number)
    
    print(f"Found {len(batch_files)} batch files to process")
    
    # Preprocess each batch file in numerical order with progress bar
    preprocess_batch_files(output_directory)
    print("All batch files processed successfully.")