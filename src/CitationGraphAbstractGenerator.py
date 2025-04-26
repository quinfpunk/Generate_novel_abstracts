import networkx as nx
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import nltk
from LinkPredictor import EnhancedLinkPredictor
try:    
    nltk.data.find('tokenizers/punkt')
except LookupError:     
    nltk.download('punkt') 
try:     
    nltk.data.find('corpora/stopwords') 
except LookupError:     
    nltk.download('stopwords') 

from sklearn.feature_extraction.text import TfidfVectorizer 
from nltk.corpus import stopwords 
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import re
import unicodedata
from sklearn.decomposition import LatentDirichletAllocation
import ast
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
from collections import defaultdict
import random
import ollama
import json 

def generate_improved_abstract(original_text, model_name="gemma3:4b", max_tokens=250):
      """
      Generate an improved academic abstract using Ollama.
      Args:
          original_text (str): The original text to create an abstract from
          model_name (str): The Ollama model to use
          max_tokens (int): Maximum length of the generated abstract
      Returns:
          str: The generated abstract
      """
      # Create an improved prompt with clear instructions and formatting
      system_prompt = """You are an expert academic researcher with extensive publication experience.
        Your task is to generate a concise, well-structured abstract based on the provided text.
        The abstract should:
        - Be approximately 150-250 words
        - Clearly state the research question/objective
        - Summarize methodology used
        - Highlight key findings/results
        - Include implications or conclusions
        - Use formal academic language and appropriate terminology
        - Be written in third person and present tense
        - Avoid citations, references, or meta-commentary
        Generate ONLY the abstract.
        """
      user_prompt = f"Generate a clear, concise academic abstract from the following text:\n\n{original_text}"
      # Configure generation parameters
      params = {
          "model": model_name,
          "messages": [
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt}
          ],
          "options": {
              "temperature": 0.3,  # Lower temperature for more focused outputs
              "top_p": 0.9,
              "max_tokens": max_tokens,
              "stop": ["---", "References", "Keywords"]  # Stop generation at these markers
          }
      }
      # Generate the abstract
      try:
          response = ollama.chat(**params)
          abstract = response['message']['content'].strip()
          return abstract
      except Exception as e:
          print(f"Error generating abstract: {e}")
          return None

class CitationGraphAbstractGenerator:
    def __init__(self, embedding_model_name='sentence-transformers/all-MiniLM-L6-v2', model_name="t5-small", linkPredictor_model_name='enhanced'):
        """
        Initialize the citation graph abstract generator

        Args:
            embedding_model: Model for embedding abstracts
            linkPredictor_model: Model for link prediction
        """
        # Download nltk resources if needed

        self.embedding_model_name = embedding_model_name
        self.linkPredictor_model_name = linkPredictor_model_name
        # use sentece transformer all-mini-LM model
        self.embedding_model = SentenceTransformer(embedding_model_name)

        self.citation_graph = nx.DiGraph()
        if linkPredictor_model_name=='enhanced':
            self.linkPredictor_model = EnhancedLinkPredictor(self.citation_graph)
        self.stop_words = list(stopwords.words('english'))

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def _update_linkPredictor(self):
        """
        Update the linkPredictor model, used when we update the grap (e.g., for new papers)
        """
        if self.linkPredictor_model_name=='enhanced':
            self.linkPredictor_model = EnhancedLinkPredictor(self.citation_graph)


    def load_citation_graph(self, papers):
        """
        Load papers and their references into a graph

        Args:
            papers: List of dictionaries, each containing 'id', 'abstract', and 'references' (list of paper ids)
        """
        # Add nodes (papers) to the graph
        for paper in papers:
            # Extract key concepts from the abstract
            key_concepts = self.extract_key_concepts(paper['abstract'])

            self.citation_graph.add_node(
                paper['id'],
                abstract=paper['abstract'],
                embedding=None,  # Will be populated later
                key_concepts=key_concepts  # Added key_concepts property
            )

        # Add edges (references)
        for paper in papers:
            for cited_paper_id in paper['references']:
                if cited_paper_id in self.citation_graph:
                    self.citation_graph.add_edge(paper['id'], cited_paper_id)

        self._update_linkPredictor()
        print(f"Graph created with {self.citation_graph.number_of_nodes()} nodes and {self.citation_graph.number_of_edges()} edges")

    def embed(self, abstract):
        print(abstract)
        inputs = self.tokenizer(abstract, return_tensors="pt", padding=True, truncation=True)
        embedding = self.model.encoder(
                      input_ids=inputs["input_ids"],
                      attention_mask=inputs["attention_mask"],
                      return_dict=True
                      )
        if embedding is None:
          print("an embedding is None")
        return embedding
    def generate_abstract_and_embedding(self, list_embeddings):
        """
        @params:
          list_embedding: list of BaseModelOutputWithPastAndCrossAttention hidden states of encoder model
        """

        max_seq_length = max([emb.last_hidden_state[-1].size(1) for emb in list_embeddings])

        # Pad each embedding's hidden states to match the maximum length
        padded_hidden_states = []
        for emb in list_embeddings:
            # Get the last hidden state
            hidden_state = emb.last_hidden_state

            # Get current dimensions
            batch_size, seq_length, hidden_dim = hidden_state.size()

            # Calculate padding required
            pad_length = max_seq_length - seq_length

            if pad_length > 0:
                # Create padding (right side only)
                padding = (0, 0, 0, pad_length)  # Format: (left, right, top, bottom)

                # Apply padding
                padded_hidden_state = F.pad(hidden_state, padding, "constant", 0)
            else:
                padded_hidden_state = hidden_state

            padded_hidden_states.append(padded_hidden_state)


        aggregated_embedding = padded_hidden_states[0]
        for l in padded_hidden_states[1:]:
          aggregated_embedding += l
        aggregated_embedding = aggregated_embedding / len(padded_hidden_states)
        aggregated_embedding = BaseModelOutputWithPastAndCrossAttentions(
            aggregated_embedding
            )

        decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id]])

        encoder_outputs = aggregated_embedding
        outputs = self.model.generate(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            max_length=len(decoder_input_ids[0]) + 250,  # Allow for a reasonably sized abstract /!\ replace with average abstract length in given graph
            num_return_sequences=1,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,

        )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the abstract part
        abstract = generated_text

        better_abstract = generate_improved_abstract(abstract)

        # in case ollama fails for any reason fallback on T5 results
        if better_abstract != None:           
            embedding = self.embed(abstract)           
            aggregated_embedding = BaseModelOutputWithPastAndCrossAttentions(             
                                                                embedding.last_hidden_state
            )           
            abstract = better_abstract         
        else:           
            embedding = aggregated_embedding 
        return abstract, aggregated_embedding

    def extract_key_concepts(self,text, score_threshold=0.1, k=7):
        """
        Extract key concepts from an abstract using TF-IDF with a score threshold.

        Args:
            text (str): Input text (e.g., an abstract).
            score_threshold (float): Minimum TF-IDF score to consider as a concept.
            k: k best selected key concepts

        Returns:
            list: List of key concepts.
        """
        # Tokenization and cleaning (removing special characters and non-alphanumeric tokens)
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Remove stop words
        filtered_tokens = [t for t in tokens if t not in self.stop_words + ['and', '-'] and len(t) > 2]  # Exclude stop words and short words

        if len(filtered_tokens) < 5:
            # If there are fewer than 5 meaningful tokens, return the tokens directly
            return list(set(filtered_tokens))

        # Apply TF-IDF to extract important terms
        tfidf = TfidfVectorizer(ngram_range=(1, 5), stop_words='english')

        try:
            tfidf_matrix = tfidf.fit_transform([text])
            feature_names = tfidf.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]

            # Pair terms with their TF-IDF scores
            scored_concepts = [(feature_names[i], scores[i]) for i in range(len(scores))]

            # Filter terms based on the score threshold
            scored_concepts = sorted(scored_concepts, key=lambda x: x[1], reverse=True)
            # Only include concepts above the threshold
            filtered_concepts = [(concept, score) for concept, score in scored_concepts if score >= score_threshold]
            filtered_concepts = [concept for concept, score in filtered_concepts[:k]]
            return filtered_concepts

        except ValueError as e:
            # If TF-IDF fails (empty vocabulary), return tokens or an error message
            print(f"Error with TF-IDF: {e}")
            return list(set(filtered_tokens))

        except Exception as e:
            print(f"TF-IDF failed: {e}")
            print(f"Tokens: {tokens}")
            print(f"Type of tokens: {type(tokens)}")
            return list(set(tokens))[:10]


    def compute_embeddings(self):
        """
        Compute and store embeddings for all papers in the graph
        """
        for node_id in tqdm(self.citation_graph.nodes(), desc="Computing embeddings"):
            abstract = self.citation_graph.nodes[node_id]['abstract']

            if not abstract:
                continue  # Skip this node if abstract is empty

            # Ensure abstract is a valid string
            if isinstance(abstract, str):
                embedding = self.embed(abstract)  # Embedding computation
                self.citation_graph.nodes[node_id]['embedding'] = embedding
            else:
                print(f"Invalid abstract format for node {node_id}: {abstract}")

        print("Embeddings computed for all nodes")

    def create_new_node(self):
        """
        Create a new node in the graph without any links

        Returns:
            new_node_id: ID of the new node
        """
        # Create new node ID
        new_node_id = f"new_paper_{len([n for n in self.citation_graph.nodes() if 'new_paper' in str(n)])}"

        # Add new node to graph without any connections
        self.citation_graph.add_node(new_node_id, abstract=None, embedding=None, key_concepts=[])

        max_iter = 10
        while self.citation_graph.nodes[new_node_id]['abstract'] is None and max_iter>0:
            predicted_links = self.predict_links(new_node_id)  # Predict links for the new node

            # Add the predicted links
            self.add_links(new_node_id, predicted_links)

            # Generate abstract based on the links
            self.generate_abstract_from_links(new_node_id)

            max_iter-=1

        # Update the target node's embedding and key concepts
        self._update_linkPredictor()

        print(f"Created new node {new_node_id} which has as reference {predicted_links} with new abstract: \n{self.citation_graph.nodes[new_node_id]['abstract']}")
        return new_node_id

    def predict_links(self, node_id):
        """
        Predict potential links for a new node using only structural information
        with stochastic elements similar to random graph generation.

        This method determines the number of links dynamically based on network properties
        and uses structural network measures with controlled randomness to select links.

        Args:
            node_id: ID of the node to predict links for

        Returns:
            predicted_links: List of node IDs that are predicted to be linked
        """

        # Get all nodes except the target and other new nodes

        predicted_links = self.linkPredictor_model.predict_links(node_id)

        return predicted_links

    def add_links(self, node_id, link_ids):
        """
        Add links from a node to other nodes

        Args:
            node_id: ID of the source node
            link_ids: List of target node IDs
        """
        for target_id in link_ids:
            if target_id in self.citation_graph:
                self.citation_graph.add_edge(node_id, target_id)

        print(f"Added {len(link_ids)} links from node {node_id}")

    def generate_abstract_from_links(self, node_id):
        """
        Generate an abstract for a node based on its links using embedding averaging

        Args:
            node_id: ID of the node to generate an abstract for
        """
        # Get linked nodes
        linked_nodes = list(self.citation_graph.successors(node_id))

        if not linked_nodes:
            print("No links found for the node")
            return None

        # Get abstracts of linked papers
        linked_abstracts = [self.citation_graph.nodes[n]['abstract'] for n in linked_nodes]

        embeddings = []
        # Create embeddings for the linked papers if they don't exist
        for n in linked_nodes:
            if self.citation_graph.nodes[n]['embedding'] is None:
                if self.citation_graph.nodes[n]['abstract'] is not None:
                  embedding = self.embed(self.citation_graph.nodes[n]['abstract'])
                  self.citation_graph.nodes[n]['embedding'] = embedding
                else:
                  linked_nodes.remove(n)

        # Get embeddings of linked papers
        linked_embeddings = [self.citation_graph.nodes[n]['embedding'] for n in linked_nodes]

        # Create an aggregated embedding for the new node and the new abstract
        abstract, aggregated_embedding = self.generate_abstract_and_embedding(linked_embeddings)

        # Store new aggregated embedding
        self.citation_graph.nodes[node_id]['embedding'] = aggregated_embedding

        # Store new abstract
        self.citation_graph.nodes[node_id]['abstract'] = abstract

        # Extract key concepts from the abstract
        key_concepts = self.extract_key_concepts(abstract)
        self.citation_graph.nodes[node_id]['key_concepts'] = key_concepts

        print(f"Generated abstract with {len(key_concepts)} key concepts: {', '.join(key_concepts)}")


    def calculate_concept_similarity(self, concepts1, concepts2, threshold=0.7):
        """
        Calculate similarity between two sets of concepts

        Args:
            concepts1: First list of concepts
            concepts2: Second list of concepts
            threshold: Similarity threshold for considering concepts as matching

        Returns:
            float: Percentage of concepts1 that have a match in concepts2
        """
        if not concepts1 or not concepts2:
            return 0.0

        # Encode concepts to get embeddings
        embeddings1 = self.embedding_model.encode(concepts1)
        embeddings2 = self.embedding_model.encode(concepts2)

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)

        # Count concepts in concepts1 that have a match above threshold in concepts2
        matched_concepts = 0
        for i in range(len(concepts1)):
            if any(similarity_matrix[i, j] > threshold for j in range(len(concepts2))):
                matched_concepts += 1

        # Return percentage
        return matched_concepts / len(concepts1)

    def evaluate_node_concepts(self, node_id, threshold=0.7):
        """
        Evaluate the concepts of a node against its linked nodes

        Args:
            node_id: ID of the node to evaluate
            threshold: Similarity threshold for considering concepts as matching

        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Get node concepts
        node_concepts = self.citation_graph.nodes[node_id]['key_concepts']

        if not node_concepts:
            print(f"No concepts found for node {node_id}")
            return None

        # Get linked nodes
        linked_nodes = list(self.citation_graph.successors(node_id))

        if not linked_nodes:
            print(f"No links found for node {node_id}")
            return None

        # Calculate concept similarity for each linked node
        similarity_percentages = []
        all_linked_concepts = set()

        for linked_id in linked_nodes:
            linked_concepts = self.citation_graph.nodes[linked_id]['key_concepts']
            all_linked_concepts.update(linked_concepts)

            similarity = self.calculate_concept_similarity(node_concepts, linked_concepts, threshold)
            similarity_percentages.append(similarity)

        # could use clasical embedding models
        # Calculate new concepts (concepts not in any linked node)
        new_concepts = []
        for concept in node_concepts:
            # Encode concept
            concept_embedding = self.embedding_model.encode([concept])[0]

            # Check if the concept exists in any linked node
            is_new = True
            for linked_id in linked_nodes:
                linked_concepts = self.citation_graph.nodes[linked_id]['key_concepts']
                if not linked_concepts:
                    continue

                linked_embeddings = self.embedding_model.encode(linked_concepts)
                similarities = cosine_similarity([concept_embedding], linked_embeddings)[0]

                if any(sim > threshold for sim in similarities):
                    is_new = False
                    break

            if is_new:
                new_concepts.append(concept)

        # Calculate metrics
        metrics = {
            'avg_concept_similarity': np.mean(similarity_percentages) if similarity_percentages else 0.0,
            'new_concepts_count': len(new_concepts),
            'new_concepts_percentage': len(new_concepts) / len(node_concepts) if node_concepts else 0.0,
            'new_concepts': new_concepts
        }

        return metrics

    def evaluate_abstract(self, node_id):
        """
        Evaluate the generated abstract

        Args:
            node_id: ID of the node

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # Get the abstract and its embedding
        abstract = self.citation_graph.nodes[node_id]['abstract']

        if abstract is None:
            print("No abstract found for the node")
            return None

        # Compute embedding if not already done
        if self.citation_graph.nodes[node_id]['embedding'] is None:
            embedding = self.embedding_model.encode(abstract)
            self.citation_graph.nodes[node_id]['embedding'] = embedding
        else:
            embedding = self.citation_graph.nodes[node_id]['embedding']

        # Get linked nodes
        linked_nodes = list(self.citation_graph.successors(node_id))

        if not linked_nodes:
            print("No links found for the node")
            return None

        # Compute similarity with linked papers
        similarities = []
        for n in linked_nodes:
            other_embedding = self.citation_graph.nodes[n]['embedding']
            similarity = cosine_similarity([embedding.last_hidden_state[0][0].detach().numpy()], [other_embedding.last_hidden_state[0][0].detach().numpy()])[0][0]
            similarities.append(similarity)

        # Calculate metrics
        metrics = {
            'avg_similarity': np.mean(similarities),
            'max_similarity': np.max(similarities),
            'min_similarity': np.min(similarities),
            'std_similarity': np.std(similarities),
            'num_linked_papers': len(linked_nodes)
        }

        # Add concept evaluation metrics
        concept_metrics = self.evaluate_node_concepts(node_id)
        if concept_metrics:
            metrics.update(concept_metrics)

        return metrics

    def visualize_graph(self, highlight_node=None):
        """
        Visualize the citation graph

        Args:
            highlight_node: Node to highlight in the visualization
        """
        plt.figure(figsize=(12, 8))

        # Define node colors
        node_colors = []
        for node in self.citation_graph.nodes():
            if node == highlight_node:
                node_colors.append('red')
            elif 'new_paper' in str(node):
                node_colors.append('green')
            else:
                node_colors.append('lightblue')

        # Define node sizes based on in-degree (citation count)
        node_sizes = []
        for node in self.citation_graph.nodes():
            in_degree = self.citation_graph.in_degree(node)
            node_sizes.append(300 + in_degree * 50)

        # Create layout
        pos = nx.spring_layout(self.citation_graph, seed=42)

        # Draw the graph
        nx.draw(
            self.citation_graph,
            pos=pos,
            with_labels=True,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.7,
            arrows=True
        )

        plt.title("Citation Graph with Generated Papers")
        plt.show()
