import networkx as nx
import numpy as np
from collections import defaultdict
import random


class EnhancedLinkPredictor:
    def __init__(self, citation_graph):
        """
        Initialize the enhanced link predictor

        Args:
            citation_graph: NetworkX DiGraph object representing the citation network
        """
        self.citation_graph = citation_graph
        self.G_undirected = citation_graph.to_undirected()  # For some metrics

    def predict_links(self, node_id, top_k=None, randomness_factor=0.1):
        """
        Predict potential links for a new node using enhanced structural metrics

        Args:
            node_id: ID of the node to predict links for
            top_k: Number of links to predict (if None, determined dynamically)
            randomness_factor: Factor controlling the randomness (0-1)

        Returns:
            predicted_links: List of node IDs that are predicted to be linked
        """
        # Get all candidate nodes (excluding the node itself and any new nodes)
        candidate_nodes = [n for n in self.citation_graph.nodes()
                          if n != node_id and 'new_paper' not in str(n)]

        if not candidate_nodes:
            return []

        # If top_k is not specified, determine it dynamically
        if top_k is None:
            top_k = self._determine_link_count(node_id)

        # Ensure we don't predict more links than available candidates
        top_k = min(top_k, len(candidate_nodes))

        # Calculate combined structural scores
        combined_scores = self._calculate_structural_scores(node_id, candidate_nodes, randomness_factor)

        # Sort nodes by score and take top_k
        ranked_nodes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        predicted_links = [node for node, score in ranked_nodes[:top_k]]

        return predicted_links

    def get_all_scores(self, node_id, candidate_nodes=None, randomness_factor=0.1):
        """
        Get all prediction scores for a node

        Args:
            node_id: ID of the node
            candidate_nodes: List of candidate nodes (if None, all nodes)
            randomness_factor: Factor controlling randomness

        Returns:
            combined_scores: Dictionary with combined scores for each candidate
        """
        if candidate_nodes is None:
            candidate_nodes = [n for n in self.citation_graph.nodes()
                              if n != node_id and 'new_paper' not in str(n)]

        # Calculate combined structural scores
        return self._calculate_structural_scores(node_id, candidate_nodes, randomness_factor)

    def _determine_link_count(self, node_id):
        """
        Determine the number of links dynamically based on network properties

        Args:
            node_id: ID of the node

        Returns:
            num_links: Number of links to predict
        """
        # Calculate basic network properties
        in_degrees = dict(self.citation_graph.in_degree())
        out_degrees = dict(self.citation_graph.out_degree())

        # Get distribution statistics
        out_degree_values = list(out_degrees.values())
        avg_out_degree = np.mean(out_degree_values)
        median_out_degree = np.median(out_degree_values)
        std_out_degree = np.std(out_degree_values)

        # Calculate a base link count using a more stable estimate
        # Use median instead of mean for robustness to outliers
        base_count = median_out_degree

        # Add stochastic variation bounded by the standard deviation
        variation = np.random.normal(0, std_out_degree / 2)

        # Apply constraints to keep the count reasonable
        num_links = max(1, int(round(base_count + variation)))

        # For new nodes with no specific pattern, cap the number to avoid too many predictions
        # max_links = int(avg_out_degree * 2)
        # num_links = min(num_links, max_links)

        return num_links

    def _calculate_structural_scores(self, node_id, candidate_nodes, randomness_factor=0.1):
        """
        Calculate structural scores using multiple enhanced metrics

        Args:
            node_id: ID of the node to predict links for
            candidate_nodes: List of candidate nodes
            randomness_factor: Factor controlling random component

        Returns:
            combined_scores: Dictionary with combined scores for each candidate
        """
        scores = {}

        # Pre-compute data needed for multiple metrics
        in_degrees = dict(self.citation_graph.in_degree())
        out_degrees = dict(self.citation_graph.out_degree())

        # 1. Calculate Adamic-Adar Index (for common neighbors)
        adamic_adar_scores = self._adamic_adar_index(node_id, candidate_nodes)

        # 2. Calculate Jaccard Coefficient
        jaccard_scores = self._jaccard_coefficient(node_id, candidate_nodes)

        # 3. Calculate Preferential Attachment scores
        pref_attachment_scores = {}
        total_in_degree = sum(in_degrees.values())
        for n in candidate_nodes:
            pref_attachment_scores[n] = in_degrees.get(n, 0) / total_in_degree if total_in_degree > 0 else 0

        # 4. Calculate Resource Allocation Index
        resource_alloc_scores = self._resource_allocation_index(node_id, candidate_nodes)

        # 5. Calculate Node Centrality
        centrality_scores = {}
        try:
            # Try to use eigenvector centrality
            eigenvector_centrality = nx.eigenvector_centrality_numpy(self.citation_graph)
            for n in candidate_nodes:
                centrality_scores[n] = eigenvector_centrality.get(n, 0)
        except:
            # Fall back to simpler degree centrality
            degree_centrality = nx.degree_centrality(self.citation_graph)
            for n in candidate_nodes:
                centrality_scores[n] = degree_centrality.get(n, 0)

        # 6. Calculate Community-based scores
        community_scores = self._community_based_scores(candidate_nodes)

        # 7. Calculate recency score if year attribute exists
        recency_scores = {}
        current_year = 2025
        for n in candidate_nodes:
            if 'year' in self.citation_graph.nodes[n]:
                year = self.citation_graph.nodes[n]['year']
                # Exponential decay instead of linear
                recency_scores[n] = np.exp(-0.25 * max(0, current_year - year))
            else:
                recency_scores[n] = 0.5  # Default value

        ppr_scores = self._personalized_pagerank(node_id, candidate_nodes)

        # 8. Random component for exploration
        random_scores = {n: np.random.random() for n in candidate_nodes}

        # Normalize all scores to 0-1 range
        normalized_scores = {
            'adamic_adar': self._normalize_scores(adamic_adar_scores),
            'jaccard': self._normalize_scores(jaccard_scores),
            'pref_attachment': self._normalize_scores(pref_attachment_scores),
            'resource_alloc': self._normalize_scores(resource_alloc_scores),
            'centrality': self._normalize_scores(centrality_scores),
            'community': self._normalize_scores(community_scores),
            'recency': self._normalize_scores(recency_scores),
            'ppr': self._normalize_scores(ppr_scores),
            'random': random_scores
        }

        # Combine scores with appropriate weights
        combined_scores = {}
        for n in candidate_nodes:
            # Structural component - different weights for different metrics
            # centrality not best but not bad
            # pref attach very bad -> so much impact
            # community bad (0.02)
            # recency bad
            # resource_alloc not bad but not best (0.15)
            # jaccard not best but not bad with pushing distrib towards right
            # academic adar not best but not bad
            structural_score = (
                0.05 * normalized_scores['adamic_adar'].get(n, 0) +
                0.15 * normalized_scores['jaccard'].get(n, 0) +
                0.02 * normalized_scores['pref_attachment'].get(n, 0) +
                0.05 * normalized_scores['resource_alloc'].get(n, 0) +
                0.03 * normalized_scores['centrality'].get(n, 0) +
                0.70 * normalized_scores['ppr'].get(n, 0)
            )
            """+
                0.05 * normalized_scores['community'].get(n, 0) +
                0.05 * normalized_scores['recency'].get(n, 0)"""
            # Combined score with randomness factor
            combined_scores[n] = (
                (1 - randomness_factor) * structural_score +
                randomness_factor * normalized_scores['random'].get(n, 0)
            )

        return combined_scores

    def _normalize_scores(self, scores):
        """Normalize scores to 0-1 range"""
        if not scores:
            return {}

        values = list(scores.values())
        min_val = min(values)
        range_val = max(values) - min_val

        if range_val == 0:
            return {k: 1.0 if v > 0 else 0.0 for k, v in scores.items()}

        return {k: (v - min_val) / range_val for k, v in scores.items()}

    def _personalized_pagerank(self, node_id, candidate_nodes, alpha=0.85, max_iter=100):
        """
        Calculate personalized PageRank scores with respect to node_id

        Args:
            node_id: Target node for personalizing PageRank
            candidate_nodes: List of candidate nodes to calculate scores for
            alpha: Damping factor (typically 0.85)
            max_iter: Maximum iterations for PageRank calculation

        Returns:
            Dictionary of nodes and their personalized PageRank scores
        """
        # Create personalization vector focused on the target node
        personalization = {n: 0.0 for n in self.citation_graph.nodes()}

        # Set high probability for the target node
        personalization[node_id] = 1.0

        try:
            # Calculate personalized PageRank
            pagerank_scores = nx.pagerank(
                self.citation_graph,
                alpha=alpha,
                personalization=personalization,
                max_iter=max_iter
            )

            # Extract scores for candidate nodes
            scores = {n: pagerank_scores.get(n, 0.0) for n in candidate_nodes}
            return scores

        except Exception as e:
            print(f"Error calculating personalized PageRank: {e}")
            # Fallback: return uniform scores
            return {n: 1.0/len(candidate_nodes) for n in candidate_nodes}

    def _adamic_adar_index(self, node_id, candidate_nodes):
        """
        Calculate Adamic-Adar index for common neighbors

        In directed graphs, we consider predecessors and successors
        """
        scores = {}

        # For each candidate node, get its neighbors
        for candidate in candidate_nodes:
            # Get neighbors
            neighbors_source = set(self.citation_graph.predecessors(node_id)) | set(self.citation_graph.successors(node_id))
            neighbors_target = set(self.citation_graph.predecessors(candidate)) | set(self.citation_graph.successors(candidate))

            # Find common neighbors
            common_neighbors = neighbors_source.intersection(neighbors_target)

            if not common_neighbors:
                scores[candidate] = 0
                continue

            # Calculate Adamic-Adar score
            score = 0
            for neighbor in common_neighbors:
                # Get degree of common neighbor
                degree = self.citation_graph.in_degree(neighbor) + self.citation_graph.out_degree(neighbor)
                if degree > 1:  # Avoid division by zero or log(1)=0
                    score += 1.0 / np.log(degree)

            scores[candidate] = score

        return scores

    def _jaccard_coefficient(self, node_id, candidate_nodes):
        """
        Calculate Jaccard similarity coefficient for common neighbors
        """
        scores = {}

        # For each candidate node, get its neighbors
        for candidate in candidate_nodes:
            if candidate == node_id:
                continue

            # Get neighbors
            neighbors_source = set(self.citation_graph.predecessors(node_id)) | set(self.citation_graph.successors(node_id))
            neighbors_target = set(self.citation_graph.predecessors(candidate)) | set(self.citation_graph.successors(candidate))

            # Calculate Jaccard coefficient
            union_size = len(neighbors_source.union(neighbors_target))
            if union_size == 0:
                scores[candidate] = 0
            else:
                intersection_size = len(neighbors_source.intersection(neighbors_target))
                scores[candidate] = intersection_size / union_size

        return scores

    def _resource_allocation_index(self, node_id, candidate_nodes):
        """
        Calculate Resource Allocation Index
        """
        scores = {}

        # For each candidate node, get its neighbors
        for candidate in candidate_nodes:
            # Get neighbors
            neighbors_source = set(self.citation_graph.predecessors(node_id)) | set(self.citation_graph.successors(node_id))
            neighbors_target = set(self.citation_graph.predecessors(candidate)) | set(self.citation_graph.successors(candidate))

            # Find common neighbors
            common_neighbors = neighbors_source.intersection(neighbors_target)

            if not common_neighbors:
                scores[candidate] = 0
                continue

            # Calculate Resource Allocation score
            score = 0
            for neighbor in common_neighbors:
                # Get degree of common neighbor
                degree = self.citation_graph.in_degree(neighbor) + self.citation_graph.out_degree(neighbor)
                if degree > 0:  # Avoid division by zero
                    score += 1.0 / degree

            scores[candidate] = score

        return scores


    def _louvain_communities(self):
        """
        Detect communities using the Louvain algorithm from networkx.
        Returns a dictionary mapping each node to a community ID.
        """
        G_undirected = self.citation_graph.to_undirected()

        # Use Louvain community detection from networkx
        communities = nx.community.louvain_communities(G_undirected)

        # Create a mapping of node to community ID
        community_map = {}
        for i, community in enumerate(communities):
            for node in community:
                community_map[node] = i

        return community_map


    def _community_based_scores(self, candidate_nodes):
        """
        Compute community-based scores.

        For large graphs, use simple approximation instead of community detection.
        For small graphs, use Louvain community detection.
        """
        scores = {}

        if len(self.citation_graph) > 5000:
            # Use degree as a proxy for community structure
            in_degrees = dict(self.citation_graph.in_degree())
            out_degrees = dict(self.citation_graph.out_degree())

            degree_buckets = defaultdict(list)
            for node in candidate_nodes:
                bucket = (in_degrees.get(node, 0) // 3, out_degrees.get(node, 0) // 3)
                degree_buckets[bucket].append(node)

            max_bucket_size = max(len(nodes) for nodes in degree_buckets.values()) if degree_buckets else 1

            for node in candidate_nodes:
                bucket = (in_degrees.get(node, 0) // 3, out_degrees.get(node, 0) // 3)
                scores[node] = len(degree_buckets[bucket]) / max_bucket_size

        else:
            try:
                communities = self._louvain_communities()

                # Calculate community sizes
                community_sizes = defaultdict(int)
                for node, cid in communities.items():
                    community_sizes[cid] += 1

                max_community_size = max(community_sizes.values()) if community_sizes else 1

                for node in candidate_nodes:
                    cid = communities.get(node)
                    if cid is not None:
                        scores[node] = community_sizes[cid] / max_community_size
                    else:
                        scores[node] = 0.1  # Fallback score for unknown nodes

            except Exception as e:
                print(f"Community detection failed: {e}. Using uniform scores.")
                for node in candidate_nodes:
                    scores[node] = 0.5

        return scores


    def analyze_network(self):
        """
        Perform network analysis on a given graph
        
        Returns:
            Dictionary containing network metrics
        """
        metrics = {}
        
        # Basic network analysis
        metrics['num_nodes'] = self.citation_graph.number_of_nodes()
        metrics['num_edges'] = self.citation_graph.number_of_edges()
        metrics['density'] = nx.density(self.citation_graph)
        
        # Degree distribution analysis
        in_degrees = [d for n, d in self.citation_graph.in_degree()]
        out_degrees = [d for n, d in self.citation_graph.out_degree()]

        metrics['avg_in_degree'] = np.mean(in_degrees)
        metrics['avg_out_degree'] = np.mean(out_degrees)
        metrics['max_in_degree'] = max(in_degrees)
        metrics['max_out_degree'] = max(out_degrees)
        
        # Connected components (directed graph)
        # Weakly connected components
        weak_cc = list(nx.weakly_connected_components(self.citation_graph))
        sizes_weak = [len(c) for c in weak_cc]
        metrics['num_wccs'] = len(weak_cc)
        metrics['largest_wcc_size'] = max(sizes_weak)
        metrics['largest_wcc_ratio'] = metrics['largest_wcc_size'] / metrics['num_nodes']
        
        # Strongly connected components
        strong_cc = list(nx.strongly_connected_components(self.citation_graph))
        sizes_strong = [len(c) for c in strong_cc]
        metrics['num_sccs'] = len(strong_cc)
        metrics['largest_scc_size'] = max(sizes_strong)
        metrics['largest_scc_ratio'] = metrics['largest_scc_size'] / metrics['num_nodes']
        
        # Clustering & transitivity (on undirected projection)
        ug = self.citation_graph.to_undirected()
        try:
            metrics['avg_clustering'] = nx.average_clustering(ug)
        except Exception:
            metrics['avg_clustering'] = None
            
        metrics['transitivity'] = nx.transitivity(ug)

        # Assortativity & reciprocity
        try:
            metrics['assortativity'] = nx.degree_assortativity_coefficient(self.citation_graph)
        except Exception:
            metrics['assortativity'] = None
            
        if self.citation_graph.is_directed():
            metrics['reciprocity'] = nx.reciprocity(self.citation_graph)
        else:
            metrics['reciprocity'] = None

        # Path‐length measures on the largest weak component
        largest_wcc_nodes = max(weak_cc, key=len)
        G_giant = ug.subgraph(largest_wcc_nodes)
        if nx.is_connected(G_giant):
            metrics['diameter'] = nx.diameter(G_giant)
            metrics['avg_path_length'] = nx.average_shortest_path_length(G_giant)
        else:
            metrics['diameter'] = None
            metrics['avg_path_length'] = None
        
        return metrics


    def evaluate(self, test_nodes=None, test_ratio=0.2, num_test_nodes=30, random_seed=42):
        """
        Evaluate link prediction performance and analyze network changes
        
        Args:
            test_nodes: Specific nodes to test (if None, randomly selected)
            test_ratio: Ratio of edges to hide for testing
            num_test_nodes: Number of nodes to test if test_nodes is None
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Get original network metrics
        original_metrics = self.analyze_network()
        
        # Print basic statistics for original network
        print("\nORIGINAL NETWORK BASIC STATISTICS:")
        print(f"Number of nodes: {original_metrics['num_nodes']}")
        print(f"Number of edges: {original_metrics['num_edges']}")
        print(f"Network density: {original_metrics['density']:.6f}")
        
        # Get nodes with sufficient outgoing edges for testing
        min_edges = 3
        potential_nodes = [n for n in self.citation_graph.nodes()
                        if self.citation_graph.out_degree(n) >= min_edges]
        
        if len(potential_nodes) < 5:
            print("Not enough nodes with sufficient edges for testing")
            return {}
        
        # Select test nodes
        if test_nodes is None:
            if len(potential_nodes) < num_test_nodes:
                test_nodes = potential_nodes
            else:
                test_nodes = random.sample(potential_nodes, num_test_nodes)
        
        # Create a copy of the graph to avoid modifying the original
        masked_graph = self.citation_graph.copy()
        total_removed_edges = 0
        removed_edge_details = []
        
        # For each test node, hide edges
        for node in test_nodes:
            # Get outgoing edges
            outgoing_edges = list(masked_graph.successors(node))
            
            # Skip if too few edges
            if len(outgoing_edges) < 2:
                continue
            
            # Hide a portion of edges for testing
            num_test_edges = max(1, int(len(outgoing_edges) * test_ratio))
            test_edges = random.sample(outgoing_edges, num_test_edges)
            
            # Remove test edges temporarily
            for target in test_edges:
                if masked_graph.has_edge(node, target):
                    masked_graph.remove_edge(node, target)
                    total_removed_edges += 1
                    removed_edge_details.append((node, target))
        
        # Analyze the masked network
        masked_metrics = EnhancedLinkPredictor(masked_graph).analyze_network()
        
        # Create a temporary predictor with the modified graph
        temp_predictor = EnhancedLinkPredictor(masked_graph)
        
        # Use link prediction to restore edges
        predicted_graph = masked_graph.copy()
        # correct_predictions = 0
        
        for node, target in removed_edge_details:
            # Predict links for this node
            predicted_links = temp_predictor.predict_links(node)
            
            # Add predicted edges
            for predicted_target in predicted_links:
                if not predicted_graph.has_edge(node, predicted_target):
                    predicted_graph.add_edge(node, predicted_target)
                
                # Check if this prediction correctly restored an edge
                # if predicted_target == target:
                #     correct_predictions += 1
        
        # Analyze the predicted network
        predicted_metrics = EnhancedLinkPredictor(predicted_graph).analyze_network()
        
        # Create comparison table
        print("\n" + "="*80)
        print(f"{'NETWORK COMPARISON TABLE':^80}")
        print("="*80)
        print(f"{'Metric':<30} | {'Original':^15} | {'Masked':^15} | {'Predicted':^15}")
        print("-"*80)
        
        # List of metrics to include in the table based on your interests
        table_metrics = [
            ('Average in-degree', 'avg_in_degree'),
            ('Average out-degree', 'avg_out_degree'),
            ('Max in-degree', 'max_in_degree'),
            ('Max out-degree', 'max_out_degree'),
            ('Number of WCCs', 'num_wccs'),
            ('Largest WCC size', 'largest_wcc_size'),
            ('Largest WCC size (%)', 'largest_wcc_ratio'),
            ('Number of SCCs', 'num_sccs'),
            ('Largest SCC size', 'largest_scc_size'),
            ('Largest SCC size (%)', 'largest_scc_ratio'),
            ('Avg clustering coef', 'avg_clustering'),
            ('Transitivity', 'transitivity'),
            ('Assortativity', 'assortativity'),
            ('Reciprocity', 'reciprocity'),
            ('Diameter', 'diameter'),
            ('Avg path length', 'avg_path_length')
        ]
        
        for label, key in table_metrics:
            # Skip metrics not calculated for original
            if key not in original_metrics:
                continue
                
            # Format each value appropriately
            o_val = original_metrics.get(key)
            m_val = masked_metrics.get(key)
            p_val = predicted_metrics.get(key)
            
            # Format based on value type
            if key.endswith('_ratio'):
                # Format as percentage
                o_str = f"{o_val:.2%}" if o_val is not None else "N/A"
                m_str = f"{m_val:.2%}" if m_val is not None else "N/A"
                p_str = f"{p_val:.2%}" if p_val is not None else "N/A"
            elif isinstance(o_val, float) or isinstance(m_val, float) or isinstance(p_val, float):
                # Format as float
                o_str = f"{o_val:.4f}" if o_val is not None else "N/A"
                m_str = f"{m_val:.4f}" if m_val is not None else "N/A"
                p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
            else:
                # Format as is
                o_str = str(o_val) if o_val is not None else "N/A"
                m_str = str(m_val) if m_val is not None else "N/A"
                p_str = str(p_val) if p_val is not None else "N/A"
            
            print(f"{label:<30} | {o_str:^15} | {m_str:^15} | {p_str:^15}")
        
        print("-"*80)
        # print(f"Link prediction accuracy: {correct_predictions}/{total_removed_edges} = {correct_predictions/max(1, total_removed_edges):.2%}")
        # print("="*80)
        
        # Return metrics
        return {
            'original': original_metrics,
            'masked': masked_metrics,
            'predicted': predicted_metrics,
            'removed_edges': total_removed_edges,
            # 'correct_predictions': correct_predictions,
            # 'accuracy': correct_predictions / max(1, total_removed_edges)
        }