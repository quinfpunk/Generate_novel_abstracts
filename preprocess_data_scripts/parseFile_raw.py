from tqdm import tqdm
import gc
import ijson
import time
import os
import numpy as np
import pandas as pd

def process_json_in_batches(file_path, batch_size=100_000, output_folder='dblp_processing'):
    """Process a large JSON file using ijson with batch processing to manage memory"""
    start = time.process_time()
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    # Get file size for progress tracking
    file_size = os.path.getsize(file_path)
    
    print(f"Processing file: {file_path} ({file_size / (1024*1024*1024):.2f} GB)")
    print(f"Using batch size of {batch_size} papers")
    print(f"Output will be saved in folder: {output_folder}")
    
    # Process in batches
    papers_batches = []
    total_papers = 0
    
    with open(file_path, 'rb') as f:  # Open in binary mode
        # Create progress bar
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Parsing JSON")
        last_pos = f.tell()
        
        # Process items in batches
        batch_count = 0
        current_batch = []
        
        for item in ijson.items(f, 'item'):
            # Update progress bar
            current_pos = f.tell()
            pbar.update(current_pos - last_pos)
            last_pos = current_pos
            
            # Extract relevant fields
            paper = {}
            paper['id'] = item.get('id')
            paper['title'] = item.get('title')
            
            year = item.get('year')
            paper['year'] = year if year else np.nan
            
            indexed_abstract = item.get('indexed_abstract')
            if indexed_abstract:
                paper['indexedAbstract'] = indexed_abstract
            else:
                paper['indexedAbstract'] = {"IndexLength": 0, "InvertedIndex": {}}
            
            n_citation = item.get('n_citation')
            paper['n_citation'] = n_citation if n_citation else 0
            
            doc_type = item.get('doc_type')
            paper['doc_type'] = doc_type if doc_type else np.nan
            
            references = item.get('references', [])
            if references:
                paper['reference_count'] = len(references)
                paper['references'] = ';'.join([str(int(r)) for r in references])
            else:
                paper['reference_count'] = np.nan
                paper['references'] = np.nan
            
            doi = item.get('doi')
            paper['doi'] = f"https://doi.org/{doi}" if doi else np.nan
            
            current_batch.append(paper)
            
            # If batch is full, save it
            if len(current_batch) >= batch_size:
                batch_count += 1
                # print(f"\nProcessing batch {batch_count}...")
                
                # Save batch to disk in the output folder
                batch_df = pd.DataFrame(current_batch)
                batch_file = os.path.join(output_folder, f"papers_batch_{batch_count}.csv")
                batch_df.to_csv(batch_file, index=False, sep=';')
                
                # Track total papers and add reference to this batch
                total_papers += len(current_batch)
                papers_batches.append(batch_file)
                
                # print(f"Saved batch {batch_count} with {len(current_batch)} papers to {batch_file}")
                # print(f"Total papers processed so far: {total_papers}")
                
                # Clear batch and free memory
                current_batch = []
                batch_df = None
                gc.collect()
        
        # Process any remaining papers in the last batch
        if current_batch:
            batch_count += 1
            batch_df = pd.DataFrame(current_batch)
            batch_file = os.path.join(output_folder, f"papers_batch_{batch_count}.csv")
            batch_df.to_csv(batch_file, index=False, sep=';')
            
            total_papers += len(current_batch)
            papers_batches.append(batch_file)
            
            # print(f"\nSaved final batch {batch_count} with {len(current_batch)} papers to {batch_file}")
        
        pbar.close()
    
    print(f"Total papers processed: {total_papers}")



if __name__ == "__main__":
    file_path = '../data/dblp_raw/dblp.v12.json'
    output_folder = '../data/dblp_batched'  # Subfolder for all output files

    # Process in batches of 100,000 papers to manage memory
    process_json_in_batches(file_path, batch_size=100_000, output_folder=output_folder)