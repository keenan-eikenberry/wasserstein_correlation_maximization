import os 
import torch
import numpy as np
from ripser import ripser
from persim import plot_diagrams, bottleneck
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm.auto import tqdm 

class EvaluateStructure:
    def __init__(self, 
                 features,
                 encoder, 
                 testloader, 
                 total_samples=None,
                 num_samples_per_class=None,
                 #Persistence analysis parameters
                 top_max_dim=2,
                 top_thresh=None,
                 top_coeff=2, 
                 #Spectral analysis parameters
                 spectral_k_values=[10, 20, 50],
                 num_eigvals=50, 
                 #Heat kernel analysis parameters
                 heat_k_values=[10, 20, 50],  
                 diffusion_times=np.logspace(-1, 1, 10),
                 #t-SNE parameters 
                 total_samples_tsne=None,
                 num_samples_per_class_tsne=None,
                 run_input_tsne=False, 
                 tsne_perplexity=30, 
                 tsne_learning_rate=200,
                 tsne_max_iter=2000,
                 tsne_metric='euclidean',
                 data_filters=None,  
                 seed=42,
                 evaluate_dir='default_evaluate_dir',
                 experiment_name='default_experiment_name'):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder
        self.features = features
        self.testloader = testloader
        if total_samples is None or total_samples > len(testloader.dataset) and num_samples_per_class is None:
            self.total_samples = len(testloader.dataset)
        else:
            self.total_samples = total_samples
        self.num_samples_per_class = num_samples_per_class
        self.top_max_dim = top_max_dim
        self.top_thresh = top_thresh
        self.top_coeff = top_coeff
        self.spectral_k_values = spectral_k_values
        self.num_eigvals = num_eigvals
        self.heat_k_values = heat_k_values
        self.diffusion_times = diffusion_times
        self.total_samples_tsne = total_samples_tsne
        self.num_samples_per_class_tsne = num_samples_per_class_tsne
        self.run_input_tsne = run_input_tsne
        self.tsne_perplexity = tsne_perplexity
        self.tsne_learning_rate = tsne_learning_rate
        self.tsne_max_iter = tsne_max_iter
        self.tsne_metric = tsne_metric
        self.data_filters = data_filters
        self.seed = seed
        self.evaluate_dir = os.path.join(evaluate_dir, 'structure_tests')
        os.makedirs(self.evaluate_dir, exist_ok=True)  
        self.experiment_name = experiment_name

        # Set random seed
        np.random.seed(self.seed)
        
        self._prepare_datasets()

        if not self.total_samples_tsne == self.total_samples or not self.num_samples_per_class_tsne == self.num_samples_per_class:
            self._prepare_tsne_datasets()
        else: 
            self.encoded_data_tsne = self.encoded_data
            self.input_data_tsne = self.input_data
            self.labels_tsne = self.labels
    
    def _create_class_mapping(self, dataset):
        if hasattr(dataset, 'classes'):
            class_names = dataset.classes
            class_mapping = {class_name: idx for idx, class_name in enumerate(class_names)}
            return class_mapping
        else:
            raise ValueError("Dataset does not have a 'classes' attribute.")

    def _prepare_datasets(self):
        """Prepare input and encoded datasets. If feature network exists, input will be features(data) and encoded will be encoder(features(data))."""
        input_data = []
        encoded_data = []
        labels = []
        
        self.has_labels = isinstance(next(iter(self.testloader)), (tuple, list))
        if self.has_labels:
            if hasattr(self.testloader.dataset, 'classes'):
                self.num_classes = len(self.testloader.dataset.classes)
                if isinstance(self.testloader.dataset.classes[0], str):
                    self.class_mapping = self._create_class_mapping(self.testloader.dataset)
                else:
                    self.class_mapping = None
            else:
                unique_labels = set()
                for _, target in self.testloader:
                    unique_labels.update(target.numpy())
                self.num_classes = len(unique_labels)

        if self.num_samples_per_class:
            if not self.has_labels:
                raise ValueError("Cannot do equal sampling with unlabeled data")
            class_counts = {i: 0 for i in range(self.num_classes)}

        with torch.no_grad():
            for batch in tqdm(self.testloader, desc="Preparing datasets"):
                if isinstance(batch, (tuple, list)):
                    data, target = batch
                else:
                    data, target = batch, None
                
                data = data.to(self.device)
                
                # Stratified sampling
                if self.num_samples_per_class and target is not None:
                    keep_indices = []
                    for i, label in enumerate(target):
                        label = label.item()
                        if class_counts[label] < self.num_samples_per_class:
                            keep_indices.append(i)
                            class_counts[label] += 1
                    
                    if keep_indices:
                        data = data[keep_indices]
                        target = target[keep_indices]
                    else:
                        continue
                
                # Regular sampling
                elif self.total_samples is not None:
                    total_collected = sum(len(x) for x in encoded_data)
                    remaining = self.total_samples - total_collected
                    if remaining <= 0:
                        break
                    if data.size(0) > remaining:
                        data = data[:remaining]
                        if target is not None:
                            target = target[:remaining]
                
                # Get input representation (either raw data or features)
                if self.features is not None:
                    input_repr = self.features(data)
                    if isinstance(input_repr, tuple):
                        input_repr = input_repr[0]  
                else:
                    input_repr = data
                
                if input_repr.dim() > 2:
                    input_data.append(input_repr.flatten(1).cpu().numpy())
                else: 
                    input_data.append(input_repr.cpu().numpy())
                
                # Get encoded representation
                encoded = self.encoder(input_repr)
                if isinstance(encoded, tuple):
                    if len(encoded) == 3:
                        encoded = encoded[2]
                    elif len(encoded) == 2:
                        encoded = encoded[1]
                if encoded.dim() > 2:
                    encoded = encoded.flatten(1)
                
                encoded_data.append(encoded.cpu().numpy())
                
                if target is not None:
                    if self.class_mapping and isinstance(target[0], str):
                        target = torch.tensor([self.class_mapping[label] for label in target])
                    labels.append(target.cpu().numpy())

                # Check if we have enough samples
                if self.num_samples_per_class and all(count >= self.num_samples_per_class for count in class_counts.values()):
                    break
                elif self.total_samples and sum(len(x) for x in encoded_data) >= self.total_samples:
                    break

        self.input_data = np.concatenate(input_data, axis=0)
        self.encoded_data = np.concatenate(encoded_data, axis=0)
        self.labels = np.concatenate(labels, axis=0) if labels else None

    def _prepare_tsne_datasets(self):
        """Prepare input and encoded datasets for t-sne visualizations. If feature network exists, input will be features(data) and encoded will be encoder(features(data))."""
        input_data_tsne = []
        encoded_data_tsne = []
        labels_tsne = []

        if self.num_samples_per_class_tsne:
            if not self.has_labels:
                raise ValueError("Cannot do equal sampling with unlabeled data")
            class_counts = {i: 0 for i in range(self.num_classes)}

        with torch.no_grad():
            for batch in tqdm(self.testloader, desc="Preparing datasets"):
                if isinstance(batch, (tuple, list)):
                    data, target = batch
                else:
                    data, target = batch, None
                
                data = data.to(self.device)
                
                # Stratified sampling
                if self.num_samples_per_class_tsne and target is not None:
                    keep_indices = []
                    for i, label in enumerate(target):
                        label = label.item()
                        if class_counts[label] < self.num_samples_per_class_tsne:
                            keep_indices.append(i)
                            class_counts[label] += 1
                    
                    if keep_indices:
                        data = data[keep_indices]
                        target = target[keep_indices]
                    else:
                        continue
                
                # Regular sampling
                elif self.total_samples_tsne is not None:
                    total_collected = sum(len(x) for x in encoded_data_tsne)
                    remaining = self.total_samples_tsne - total_collected
                    if remaining <= 0:
                        break
                    if data.size(0) > remaining:
                        data = data[:remaining]
                        if target is not None:
                            target = target[:remaining]
                
                # Get input representation (either raw data or features)
                if self.features is not None:
                    input_repr = self.features(data)
                    if isinstance(input_repr, tuple):
                        input_repr = input_repr[0]  
                else:
                    input_repr = data
                
                if input_repr.dim() > 2:
                    input_data_tsne.append(input_repr.flatten(1).cpu().numpy())
                else: 
                    input_data_tsne.append(input_repr.cpu().numpy())
                
                # Get encoded representation
                encoded = self.encoder(input_repr)
                if isinstance(encoded, tuple):
                    if len(encoded) == 3:
                        encoded = encoded[2]
                    elif len(encoded) == 2:
                        encoded = encoded[1]
                if encoded.dim() > 2:
                    encoded = encoded.flatten(1)
                
                encoded_data_tsne.append(encoded.cpu().numpy())
                
                if target is not None:
                    if self.class_mapping and isinstance(target[0], str):
                        target = torch.tensor([self.class_mapping[label] for label in target])
                    labels_tsne.append(target.cpu().numpy())

                # Check if we have enough samples
                if self.num_samples_per_class_tsne and all(count >= self.num_samples_per_class_tsne for count in class_counts.values()):
                    break
                elif self.total_samples_tsne and sum(len(x) for x in encoded_data_tsne) >= self.total_samples_tsne:
                    break

        self.input_data_tsne = np.concatenate(input_data_tsne, axis=0)
        self.encoded_data_tsne = np.concatenate(encoded_data_tsne, axis=0)
        self.labels_tsne = np.concatenate(labels_tsne, axis=0) if labels_tsne else None

    def compute_persistence_diagrams(self, max_dim=2, thresh=None, coeff=2):
        """Compute persistence diagrams for input and encoded distributions.
        
        Args:
            max_dim: Maximum homology dimension to compute (default 2)
            thresh: Maximum radius for the Vietoris-Rips complex
            coeff: Field coefficients for homology computation (default 2)
        """
        # Compute pairwise distances 
        n = len(self.input_data)
        input_dists = np.zeros((n, n))
        encoded_dists = np.zeros((n, n))
        
        for i in tqdm(range(n), desc="Computing pairwise distances"):
            input_dists[i] = np.linalg.norm(self.input_data[i:i+1] - self.input_data, axis=1)
            encoded_dists[i] = np.linalg.norm(self.encoded_data[i:i+1] - self.encoded_data, axis=1)

        if thresh is None:
            input_thresh = 0.2 * np.max(input_dists)
            encoded_thresh = 0.2 * np.max(encoded_dists)
        else:
            input_thresh = encoded_thresh = thresh

        try:
            with tqdm(total=2, desc="Computing persistence diagrams") as pbar:
                # Use precomputed distance matrices
                input_dgms = ripser(
                    input_dists,
                    maxdim=max_dim,
                    thresh=input_thresh,
                    coeff=coeff,
                    distance_matrix=True
                )['dgms']
                pbar.update(1)
                
                encoded_dgms = ripser(
                    encoded_dists,
                    maxdim=max_dim,
                    thresh=encoded_thresh,
                    coeff=coeff,
                    distance_matrix=True
                )['dgms']
                pbar.update(1)
            
            return {
                'input': input_dgms,
                'encoded': encoded_dgms
            }
            
        except (MemoryError, RuntimeError) as e:
            print(f"Error during persistence computation: {e}")
            print("Try reducing sample points.")
            return None
    

    def compute_diagram_distances(self, diagrams):
        """Compute bottleneck distances between persistence diagrams."""
        distances = {}
        for dim in range(len(diagrams['input'])):
            distances[f'input_vs_encoded_dim{dim}'] = bottleneck(
                diagrams['input'][dim],
                diagrams['encoded'][dim]
            )
        return distances
    
    
    def plot_persistence_comparison(self, diagrams):
        """Plot persistence diagrams for comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        plot_diagrams(diagrams['input'], show=False, ax=ax1)
        ax1.set_title('Input Data Persistence')
        
        plot_diagrams(diagrams['encoded'], show=False, ax=ax2)
        ax2.set_title('Encoded Data Persistence')
        
        plt.tight_layout()
        return fig

    def run_topological_analysis(self, max_dim, thresh, coeff, epoch=None):
        """Run topological analysis and save results.
        
        Args:
            max_dim: Maximum homology dimension to compute
            thresh: Maximum radius for the Vietoris-Rips complex
            coeff: Field coefficients for homology computation
            epoch: Optional epoch number for saving results
        """
        print(f"\nStarting topological analysis...")
        diagrams = self.compute_persistence_diagrams(
            max_dim=max_dim,
            thresh=thresh,
            coeff=coeff
        )
        
        if diagrams is None:
            print("Persistence computation failed. Skipping remaining analysis.")
            return None
            
        results = {
            'diagrams': diagrams,
            'distances': self.compute_diagram_distances(diagrams),
            'plot': self.plot_persistence_comparison(diagrams)
        }
        
        self.save_topological_analysis(results, epoch)
        
        return results
    
    def save_topological_analysis(self, results, epoch=None):
        """Save persistence diagrams plot and bottleneck distances.
        
        Args:
            results: Dict containing 'diagrams', 'distances', and 'plot'
            epoch: Optional epoch number for filename
        """
        # Save plot
        plots_dir = os.path.join(self.evaluate_dir, 'persistence_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        epoch_str = f'epoch_{epoch}_' if epoch is not None else ''
        plot_filename = f'{epoch_str}persistence_diagrams_{self.experiment_name}.png'
        results['plot'].savefig(os.path.join(plots_dir, plot_filename), dpi=200)
        plt.close(results['plot'])
        
        # Save distances to log file
        log_dir = os.path.join(self.evaluate_dir, 'persistence_logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_filename = f'{epoch_str}bottleneck_distances_{self.experiment_name}.txt'
        with open(os.path.join(log_dir, log_filename), 'w') as f:
            f.write(f"Bottleneck distances between input and encoded spaces:\n")
            for dim, dist in results['distances'].items():
                f.write(f"{dim}: {dist:.8f}\n")
        
        print(f"Saved persistence analysis results for{' epoch ' + str(epoch) if epoch is not None else ''}")


    def compute_spectral_properties(self, data, k):
        """Compute normalized eigenvalues of the graph Laplacian plus auxiliary info
        
        Args:
            data: Input data matrix of shape (n_samples, n_features)
            k: Number of nearest neighbors for graph construction
            
        Returns:
            Dictionary containing spectral properties:
            - normalized_spectrum: Normalized eigenvalues of the graph Laplacian
            - fiedler_value: Second smallest eigenvalue (connectivity measure)
            - max_spectral_gap: Maximum difference between consecutive eigenvalues
            - trace: Sum of eigenvalues (proportional to total variation)
        """
        n_samples = data.shape[0]
        n_components = min(self.num_eigvals, n_samples - 1)
        
        # k-NN graph
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)
        _, indices = nbrs.kneighbors(data)
        
        # Sparse adjacency matrix 
        rows = np.repeat(np.arange(n_samples), k)
        cols = indices.ravel()
        ones = np.ones(rows.shape[0], dtype=np.float32)
        adj_matrix = csr_matrix((ones, (rows, cols)), shape=(n_samples, n_samples))
        
        # Make symmetric 
        adj_matrix = (adj_matrix + adj_matrix.T) > 0
        
        degree = np.array(adj_matrix.sum(axis=1)).ravel()
        
        # D and D^(-1/2)
        D = csr_matrix((degree, (range(n_samples), range(n_samples))))
        deg_sqrt_inv = 1.0 / np.sqrt(degree)
        D_sqrt_inv = csr_matrix((deg_sqrt_inv, (range(n_samples), range(n_samples))))
        
        # Normalized Laplacian L_norm = D^(-1/2) (D - A) D^(-1/2)
        L_norm = D_sqrt_inv @ (D - adj_matrix) @ D_sqrt_inv
        
        # Eigenvalues (which='SM' for smallest magnitude)
        eigenvalues, _ = eigsh(L_norm, k=n_components, which='SM')
        eigenvalues = np.sort(eigenvalues)
        
        # Normalize by largest eigenvalue
        normalized_spectrum = eigenvalues / eigenvalues.max()
        
        # Compute auxiliary spectral properties
        fiedler_value = eigenvalues[1] if len(eigenvalues) > 1 else 0
        spectral_gaps = np.diff(eigenvalues)
        max_spectral_gap = np.max(spectral_gaps) if len(spectral_gaps) > 0 else 0
        trace = np.sum(eigenvalues)
        
        return {
            'normalized_spectrum': normalized_spectrum,
            'fiedler_value': fiedler_value,
            'max_spectral_gap': max_spectral_gap,
            'trace': trace
        }


    def compute_spectral_distances(self):
        """Compute spectral distances and property differences for different k values"""
        distances = {}
        spectral_properties = {}
        property_differences = {}

        for k in tqdm(self.spectral_k_values, desc="Computing spectral distances"):
            input_props = self.compute_spectral_properties(self.input_data, k)
            encoded_props = self.compute_spectral_properties(self.encoded_data, k)
            
            spectral_properties[f'k={k}'] = {
                'input': input_props,
                'encoded': encoded_props
            }
            
            # Compute distance between normalized spectra
            distances[f'k={k}'] = np.linalg.norm(
                input_props['normalized_spectrum'] - encoded_props['normalized_spectrum']
            )
            
            # Compute absolute and relative differences for auxiliary properties
            diffs = {}
            
            # Fiedler value
            fiedler_abs_diff = abs(encoded_props['fiedler_value'] - input_props['fiedler_value'])
            fiedler_rel_diff = fiedler_abs_diff / input_props['fiedler_value'] if input_props['fiedler_value'] != 0 else float('inf')
            diffs['fiedler'] = {
                'absolute': fiedler_abs_diff,
                'relative': fiedler_rel_diff
            }
            
            # Max spectral gap
            gap_abs_diff = abs(encoded_props['max_spectral_gap'] - input_props['max_spectral_gap'])
            gap_rel_diff = gap_abs_diff / input_props['max_spectral_gap'] if input_props['max_spectral_gap'] != 0 else float('inf')
            diffs['max_spectral_gap'] = {
                'absolute': gap_abs_diff,
                'relative': gap_rel_diff
            }
            
            # Trace
            trace_abs_diff = abs(encoded_props['trace'] - input_props['trace'])
            trace_rel_diff = trace_abs_diff / input_props['trace'] if input_props['trace'] != 0 else float('inf')
            diffs['trace'] = {
                'absolute': trace_abs_diff,
                'relative': trace_rel_diff
            }
            
            property_differences[f'k={k}'] = diffs
            
        return distances, spectral_properties, property_differences


    def run_spectral_analysis(self, epoch=None):
        """Run spectral analysis and save results. Optional epoch number for filename
        """
        print(f"\nStarting spectral analysis...")
        distances, spectral_properties, property_differences = self.compute_spectral_distances()
        
        results = {
            'distances': distances,
            'spectral_properties': spectral_properties,
            'property_differences': property_differences
        }
        
        self.save_spectral_analysis(results, epoch)
        
        return results
    

    def save_spectral_analysis(self, results, epoch=None):
        """Save spectral analysis results. Optional epoch number for filename
        """
        # Save distances to log file
        log_dir = os.path.join(self.evaluate_dir, 'spectral_logs')
        os.makedirs(log_dir, exist_ok=True)
        
        epoch_str = f'epoch_{epoch}_' if epoch is not None else ''
        
        # Save main log with distances
        log_filename = f'{epoch_str}spectral_distances_{self.experiment_name}.txt'
        
        with open(os.path.join(log_dir, log_filename), 'w') as f:
            f.write("Spectral distances between input and encoded spaces:\n")
            for k, dist in results['distances'].items():
                f.write(f"{k}: {dist:.8f}\n")
            f.write(f"\nAverage distance: {np.mean(list(results['distances'].values())):.8f}")
        
        # Save auxiliary spectral properties
        aux_log_filename = f'{epoch_str}spectral_properties_{self.experiment_name}.txt'
        
        with open(os.path.join(log_dir, aux_log_filename), 'w') as f:
            f.write("Auxiliary spectral properties and differences:\n\n")
            
            for k in self.spectral_k_values:
                k_key = f'k={k}'
                input_props = results['spectral_properties'][k_key]['input']
                encoded_props = results['spectral_properties'][k_key]['encoded']
                diffs = results['property_differences'][k_key]
                
                f.write(f"Properties for k={k}:\n")
                
                # Fiedler value (connectivity)
                f.write(f"  Fiedler value (connectivity):\n")
                f.write(f"    Input: {input_props['fiedler_value']:.8f}\n")
                f.write(f"    Encoded: {encoded_props['fiedler_value']:.8f}\n")
                f.write(f"    Absolute difference: {diffs['fiedler']['absolute']:.8f}\n")
                f.write(f"    Relative difference: {diffs['fiedler']['relative']:.8f}\n")
                
                # Maximum spectral gap (cluster separation)
                f.write(f"  Maximum spectral gap (cluster separation):\n")
                f.write(f"    Input: {input_props['max_spectral_gap']:.8f}\n")
                f.write(f"    Encoded: {encoded_props['max_spectral_gap']:.8f}\n")
                f.write(f"    Absolute difference: {diffs['max_spectral_gap']['absolute']:.8f}\n")
                f.write(f"    Relative difference: {diffs['max_spectral_gap']['relative']:.8f}\n")
                
                # Trace (total variation)
                f.write(f"  Trace (total variation):\n")
                f.write(f"    Input: {input_props['trace']:.8f}\n")
                f.write(f"    Encoded: {encoded_props['trace']:.8f}\n")
                f.write(f"    Absolute difference: {diffs['trace']['absolute']:.8f}\n")
                f.write(f"    Relative difference: {diffs['trace']['relative']:.8f}\n\n")
        
        print(f"Saved spectral analysis results for{' epoch ' + str(epoch) if epoch is not None else ''}")

    def compute_heat_kernel_signature(self, data, k, diffusion_times):
        """Compute heat kernel signatures for different time points with specified k.
        
        Args:
            data: Input data matrix of shape (n_samples, n_features)
            k: Number of nearest neighbors for graph construction
            diffusion_times: Array of diffusion time points
        
        Returns:
            Heat kernel signatures for each point at each diffusion time
        """
        n_samples = data.shape[0]
        
        # kNN graph, sigma, Guassian weights
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)
        distances, indices = nbrs.kneighbors(data)
        sigma = np.mean(distances[:, -1])
        weights = np.exp(-distances**2 / (2 * sigma**2))
        
        # Weighted adjacency matrix
        rows = np.repeat(np.arange(n_samples), k)
        cols = indices.ravel()
        adj_matrix = csr_matrix((weights.ravel(), (rows, cols)), shape=(n_samples, n_samples))
        adj_matrix = (adj_matrix + adj_matrix.T) / 2  # Make symmetric
        
        degree = np.array(adj_matrix.sum(axis=1)).ravel()
        
        # D and D^(-1/2)
        D_sqrt_inv = csr_matrix((1.0 / np.sqrt(degree), (range(n_samples), range(n_samples))))
        D = csr_matrix((degree, (range(n_samples), range(n_samples))))
        
        # Normalized Laplacian L_norm = D^(-1/2) (D - A) D^(-1/2)
        L_norm = D_sqrt_inv @ (D - adj_matrix) @ D_sqrt_inv
        
        # Eigendecomposition
        n_components = min(50, n_samples - 1)
        eigenvalues, eigenvectors = eigsh(L_norm, k=n_components, which='SM')
        sort_idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]
        
        # Heat kernel signatures 
        hks = np.zeros((n_samples, len(diffusion_times)))
        for i, diffusion_time in enumerate(tqdm(diffusion_times, desc=f"Computing HKS (k={k})")):
            heat_kernel = np.exp(-eigenvalues * diffusion_time)
            hks[:, i] = np.sum(eigenvectors**2 * heat_kernel[None, :], axis=1)
        
        return hks

    def compute_hks_distances(self, k_values, diffusion_times):
        """Compute distances between heat kernel signatures for multiple k values.
        
        Args:
            k_values: List of k values for nearest neighbors 
            diffusion_times: Array of diffusion time points
        
        Returns:
            Dictionary of distances and HKS results for each k value
        """
        results = {}
        
        for k in tqdm(k_values, desc="Computing HKS for different k values"):
            print(f"\nComputing heat kernel signatures for k={k}")
            print("Processing input space...")
            input_hks = self.compute_heat_kernel_signature(self.input_data, k, diffusion_times)
            
            print("Processing encoded space...")
            encoded_hks = self.compute_heat_kernel_signature(self.encoded_data, k, diffusion_times)
            
            # Compute L2 distance for each diffusion time
            k_distances = {}
            for i, diffusion_time in enumerate(tqdm(diffusion_times, desc=f"Computing distances for k={k}")):
                # Normalize HKS values at this diffusion time
                input_norm = input_hks[:, i] / np.linalg.norm(input_hks[:, i])
                encoded_norm = encoded_hks[:, i] / np.linalg.norm(encoded_hks[:, i])
        
                k_distances[f't={diffusion_time:.8f}'] = np.linalg.norm(input_norm - encoded_norm)
            
            results[f'k={k}'] = {
                'distances': k_distances,
                'input_hks': input_hks,
                'encoded_hks': encoded_hks,
                'avg_distance': np.mean(list(k_distances.values()))
            }
        
        return results

    def plot_hks_comparison(self, input_hks, encoded_hks, diffusion_times, k=None):
        """Plot heat kernel signature comparison.
        
        Args:
            input_hks: HKS for input space
            encoded_hks: HKS for encoded space
            diffusion_times: Diffusion times used for HKS computation
            k: k value used (for title)
        """
        k_str = f" (k={k})" if k is not None else ""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot mean and std of HKS
        input_mean = input_hks.mean(axis=0)
        input_std = input_hks.std(axis=0)
        encoded_mean = encoded_hks.mean(axis=0)
        encoded_std = encoded_hks.std(axis=0)
        
        # Mean and std plot
        ax1.plot(diffusion_times, input_mean, label='Input Space', alpha=0.7)
        ax1.fill_between(diffusion_times, 
                        input_mean - input_std,
                        input_mean + input_std,
                        alpha=0.2)
        
        ax1.plot(diffusion_times, encoded_mean, label='Encoded Space', alpha=0.7)
        ax1.fill_between(diffusion_times,
                        encoded_mean - encoded_std,
                        encoded_mean + encoded_std,
                        alpha=0.2)
        
        ax1.set_xscale('log')
        ax1.set_title(f'Heat Kernel Signature Mean and Std{k_str}')
        ax1.set_xlabel('Diffusion Time')
        ax1.set_ylabel('HKS Value')
        ax1.legend()
        
        # Distribution plot for a middle diffusion time
        mid_idx = len(diffusion_times) // 2
        mid_time = diffusion_times[mid_idx]
        
        ax2.hist(input_hks[:, mid_idx], bins=50, alpha=0.5, label='Input Space',
                density=True)
        ax2.hist(encoded_hks[:, mid_idx], bins=50, alpha=0.5, label='Encoded Space',
                density=True)
        ax2.set_title(f'HKS Distribution at t={mid_time:.8f}{k_str}')
        ax2.set_xlabel('HKS Value')
        ax2.set_ylabel('Density')
        ax2.legend()
        
        plt.tight_layout()
        return fig

    def run_heat_kernel_analysis(self, k_values, diffusion_times, epoch=None):
        """Run heat kernel analysis and save results.
        
        Args:
            epoch: Optional epoch number for saving results
            k_values: List of k values for nearest neighbors 
            diffusion_times: Array of diffusion time points 
        """
        print("\nStarting heat kernel analysis...")
        
        results = self.compute_hks_distances(k_values, diffusion_times)
        
        for k_key in results:
            k = int(k_key.split('=')[1])  
            results[k_key]['plot'] = self.plot_hks_comparison(
                results[k_key]['input_hks'],
                results[k_key]['encoded_hks'],
                diffusion_times,
                k=k
            )
        
        # Overall average across k values
        overall_avg = np.mean([results[k_key]['avg_distance'] for k_key in results])
        results['overall_avg_distance'] = overall_avg
        
        # Save results
        self.save_heat_kernel_analysis(results, epoch)
        
        return results

    def save_heat_kernel_analysis(self, results, epoch=None):
        """Save heat kernel analysis results. Optional epoch number for filename
        """
        # Define directories
        log_dir = os.path.join(self.evaluate_dir, 'hks_logs')
        plots_dir = os.path.join(self.evaluate_dir, 'hks_plots')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        # Create epoch string for filenames
        epoch_str = f'epoch_{epoch}_' if epoch is not None else ''
        
        # Save distances to log file for each k
        for k_key in [k for k in results.keys() if k.startswith('k=')]:
            k = int(k_key.split('=')[1])  
            
            log_filename = f'{epoch_str}hks_distances_k{k}_{self.experiment_name}.txt'
            with open(os.path.join(log_dir, log_filename), 'w') as f:
                f.write(f"Heat kernel signature distances between input and encoded spaces (k={k}):\n")
                f.write("\nTime-wise distances:\n")
                for t, dist in results[k_key]['distances'].items():
                    f.write(f"{t}: {dist:.8f}\n")
                
                # Summary statistics
                dists = list(results[k_key]['distances'].values())
                f.write("\nSummary Statistics:")
                f.write(f"\nAverage distance: {np.mean(dists):.8f}")
                f.write(f"\nMedian distance: {np.median(dists):.8f}")
                f.write(f"\nStandard deviation: {np.std(dists):.8f}")
                f.write(f"\nMinimum distance: {np.min(dists):.8f}")
                f.write(f"\nMaximum distance: {np.max(dists):.8f}")
            
            plot_filename = f'{epoch_str}hks_comparison_k{k}_{self.experiment_name}.png'
            results[k_key]['plot'].savefig(os.path.join(plots_dir, plot_filename))
            plt.close(results[k_key]['plot'])
        
        # Save overall summary if multiple k values were used
        if len([k for k in results.keys() if k.startswith('k=')]) > 1:
            summary_filename = f'{epoch_str}hks_summary_{self.experiment_name}.txt'
            with open(os.path.join(log_dir, summary_filename), 'w') as f:
                f.write("Heat kernel signature analysis summary across different k values:\n\n")
                for k_key in sorted([k for k in results.keys() if k.startswith('k=')]):
                    k = int(k_key.split('=')[1])
                    f.write(f"k={k}: Average distance = {results[k_key]['avg_distance']:.8f}\n")
                f.write(f"\nOverall average distance: {results['overall_avg_distance']:.8f}")
        
        print(f"Saved heat kernel analysis results for{' epoch ' + str(epoch) if epoch is not None else ''}")

    def run_tsne_reduction(self):
        """Apply t-SNE dimensionality reduction to input and encoded data 
        """
        print("Running t-SNE dimensionality reduction...")
        
        # Apply t-SNE to input space
        if self.run_input_tsne:
            n_samples_input = self.input_data_tsne.shape[0]
            reducer_input = TSNE(
                n_components=2,
                perplexity=min(self.tsne_perplexity, n_samples_input - 1),
                learning_rate=self.tsne_learning_rate,
                max_iter=self.tsne_max_iter,
                random_state=self.seed
            )
            self.input_reduced = reducer_input.fit_transform(self.input_data_tsne)
        
        # Apply t-SNE to encoded space
        n_samples_encoded = self.encoded_data_tsne.shape[0]
        reducer_encoded = TSNE(
            n_components=2,
            perplexity=min(self.tsne_perplexity, n_samples_encoded - 1),
            learning_rate=self.tsne_learning_rate,
            max_iter=self.tsne_max_iter,
            random_state=self.seed
        )
        self.encoded_reduced = reducer_encoded.fit_transform(self.encoded_data_tsne)
        
        print(f"t-SNE dimensionality reduction complete.")


    def plot_tsne_visualization(self, space='input', epoch=None, data_filter=None):
        """t-SNE visualization
        
        Args:
            space: 'input' or 'encoded' to specify which space to plot
            epoch: Optional epoch number for filename
            data_filter: Optional filter for specific classes to visualize
        """
        if space == 'input':
            reduced_data = self.input_reduced
            title_prefix = 'Input'
            file_prefix = 'tsne_input'
        else:  # encoded
            reduced_data = self.encoded_reduced
            title_prefix = 'Encoded'
            file_prefix = 'tsne_encoded'
        
        # Data filtering
        if data_filter is not None and hasattr(self, 'class_mapping') and self.class_mapping:
            if not all(isinstance(item, type(data_filter[0])) for item in data_filter):
                raise ValueError("data_filter must contain all integers or all strings.")
            
            data_filter_copy = data_filter
            if isinstance(data_filter[0], str):
                data_filter = [self.class_mapping[cls_name] for cls_name in data_filter if cls_name in self.class_mapping]
        
        if data_filter is not None:
            indices = [i for i, label in enumerate(self.labels_tsne) if label in data_filter]
            if indices:
                filtered_data = reduced_data[indices]
                filtered_labels = self.labels_tsne[indices]
                unique_labels = np.unique(filtered_labels)
                for i in range(self.num_classes):
                    if i not in unique_labels:
                        filtered_labels = np.append(filtered_labels, i)
                        filtered_data = np.append(filtered_data, np.zeros((1, filtered_data.shape[1])), axis=0)
            else:
                filtered_data = reduced_data
                filtered_labels = self.labels_tsne
        else:
            filtered_data = reduced_data
            filtered_labels = self.labels_tsne
            data_filter_copy = None
        
        # Create plot
        plt.figure(figsize=(7, 6))
        if filtered_labels is not None:
            scatter = plt.scatter(filtered_data[:, 0], filtered_data[:, 1], c=filtered_labels, cmap=plt.cm.Spectral)
            plt.colorbar(scatter, ticks=range(self.num_classes))
        else:
            plt.scatter(filtered_data[:, 0], filtered_data[:, 1], cmap=plt.cm.Spectral)
        
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title(f'{title_prefix} Space (t-SNE)')
        plt.grid(True)
        
        # Filename
        epoch_str = f'epoch_{epoch}_' if epoch is not None else ''
        if data_filter is not None:
            filter_str = "_and_".join(map(lambda x: str(x).replace('/', '_'), data_filter_copy))
            filename = f'{epoch_str}{file_prefix}_{filter_str}_{self.experiment_name}.png'
        else:
            filename = f'{epoch_str}{file_prefix}_all_{self.experiment_name}.png'
        
        # Save figure
        tsne_dir = os.path.join(self.evaluate_dir, 'tsne_plots')
        os.makedirs(tsne_dir, exist_ok=True)
        plt.savefig(os.path.join(tsne_dir, filename), dpi=200)
        plt.close()
        
        print(f"Done plotting {title_prefix} t-SNE with filter {data_filter_copy if data_filter is not None else 'None'}")


    def run_tsne_visualization(self, epoch=None, data_filters=None):
        """Run t-SNE visualization and save results.
        
        Args:
            epoch: Optional epoch number for filename 
            data_filters: List of lists or a single list of classes to filter for visualization; if None, only all classes will be plotted. 
        """
        print("\nStarting t-SNE visualization...")
        
        # Run dimensionality reduction
        self.run_tsne_reduction()
        
        # Plot all classes
        if self.run_input_tsne:
            self.plot_tsne_visualization(space='input', epoch=epoch)
        self.plot_tsne_visualization(space='encoded', epoch=epoch)
        
        if data_filters is not None:
            if not isinstance(data_filters[0], list):
                data_filters = [data_filters]
            
            for data_filter in data_filters:
                if data_filter:
                    if self.run_input_tsne: 
                        self.plot_tsne_visualization(space='input', epoch=epoch, data_filter=data_filter)
                    self.plot_tsne_visualization(space='encoded', epoch=epoch, data_filter=data_filter)
        
        # Save basic metadata 
        log_dir = os.path.join(self.evaluate_dir, 'tsne_logs')
        os.makedirs(log_dir, exist_ok=True)

        log_filename = f'tsne_info_{self.experiment_name}.txt'
        log_filepath = os.path.join(log_dir, log_filename)

        if not os.path.exists(log_filepath):
            with open(log_filepath, 'w') as f:
                f.write("t-SNE Visualization Information:\n")
                f.write(f"Number of Samples: {self.input_data_tsne.shape[0]}\n")
                f.write(f"\nt-SNE Parameters:\n")
                f.write(f"  Perplexity: {self.tsne_perplexity}\n")
                f.write(f"  Learning Rate: {self.tsne_learning_rate}\n")
                f.write(f"  Max Iterations: {self.tsne_max_iter}\n")
                
                if data_filters:
                    f.write("\nData Filters Applied:\n")
                    for i, df in enumerate(data_filters):
                        f.write(f"  Filter {i}: {df}\n")
            
            print(f"Saved t-SNE metadata information")
        
        print(f"Saved t-SNE visualization for{' epoch ' + str(epoch) if epoch is not None else ''}")

    def save_dimensionality_reduction_info(self):
        """
        Save information about dimensionality reduction 
        """
        log_dir = os.path.join(self.evaluate_dir, 'dimensionality_reduction')
        os.makedirs(log_dir, exist_ok=True)

        filename = f'dimensionality_reduction_{self.experiment_name}.txt'
        filepath = os.path.join(log_dir, filename)
        
        if os.path.exists(filepath):
            return
        
        with open(filepath, 'w') as f:
            f.write("Dimensionality Reduction Information:\n")
            f.write(f"\nInput dimension: {self.input_data.shape[1]}")
            f.write(f"\nEncoded dimension: {self.encoded_data.shape[1]}")
            f.write(f"\nDimensionality reduction ratio: {self.input_data.shape[1]/self.encoded_data.shape[1]:.2f}")
            
        print(f"Saved dimensionality reduction information")


    def run_tests(self, tests=['topological', 'spectral', 'heat', 'tsne'], epoch=None):
        """Run selected tests/visualizations.
        
        Args:
            tests: List of tests to run. Options: 'topological', 'spectral', 'heat', 'tsne'
            epoch: Optional epoch number for saving results
        
        Returns:
            Dictionary with results from each requested test
        """
        results = {}
        print(f"\nRunning tests for: {', '.join(tests)}")
        
        # Save dimensionality reduction info 
        self.save_dimensionality_reduction_info()
        
        # Run selected tests
        if 'topological' in tests:
            topological_results = self.run_topological_analysis(
                max_dim=self.top_max_dim, 
                thresh=self.top_thresh, 
                coeff=self.top_coeff,
                epoch=epoch
            )
            results['topological'] = topological_results
        
        if 'spectral' in tests:
            spectral_results = self.run_spectral_analysis(epoch=epoch)
            results['spectral'] = spectral_results
        
        if 'heat' in tests:
            heat_results = self.run_heat_kernel_analysis(
                epoch=epoch,
                k_values=self.heat_k_values,
                diffusion_times=self.diffusion_times
            )
            results['heat'] = heat_results
        
        if 'tsne_all' in tests:
            self.run_input_tsne = True
            self.run_tsne_visualization(
                epoch=epoch,
                data_filters=self.data_filters
            )
        
        if 'tsne' in tests: 
            self.run_tsne_visualization(
                epoch=epoch,
                data_filters=self.data_filters
            )

        # Results Summary
        summary_report = os.path.join(self.evaluate_dir, f"summary_report_epoch_{epoch}.txt")
        with open(summary_report, 'w') as f:
            f.write(f"Structure Preservation Tests {self.experiment_name} at epoch {epoch}\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Dimension reduction: {self.input_data.shape[1]} → {self.encoded_data.shape[1]} \n" 
                f"({self.input_data.shape[1]/self.encoded_data.shape[1]:.2f}x)\n")
            
            f.write("\n")

            if 'topological' in tests and results['topological']:
                topo_dists = results['topological']['distances']
                f.write("\nTopological (Bottleneck) distances:\n")
                for dim, dist in topo_dists.items():
                    f.write(f"  {dim}: {dist:.8f}\n")
            
            f.write("\n")

            if 'spectral' in tests and results['spectral']:
                spec_dists = results['spectral']['distances']
                f.write("\nSpectral distances:\n")
                for k, dist in spec_dists.items():
                    f.write(f"  {k}: {dist:.8f}\n")
                f.write(f"  Average: {np.mean(list(spec_dists.values())):.8f}\n")

            f.write("\n")    
            
            if 'heat' in tests and results['heat']:
                f.write("\nHeat kernel distances (averaged across time):\n")
                # Average distance across diffusion times for each k value
                for k_key in sorted([k for k in results['heat'].keys() if k.startswith('k=')]):
                    avg_dist = results['heat'][k_key]['avg_distance']
                    f.write(f"  {k_key}: {avg_dist:.8f}\n")
            
                # Overall average
                heat_avg = results['heat'].get('overall_avg_distance', 0)
                f.write(f"  Overall average: {heat_avg:.8f}\n")

            f.write("\n")

            print(f"\nAll results saved to: {self.evaluate_dir}")

        return results