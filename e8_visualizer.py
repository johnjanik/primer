import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import pickle

class E8Visualizer:
    """Visualization tools for E8 prime decoding results"""
    
    def __init__(self, results_file="e8_decoding_results.pkl"):
        with open(results_file, 'rb') as f:
            data = pickle.load(f)
        
        self.results = data['results']
        self.triggers = data['triggers']
        self.blocks = data['blocks']
        
    def plot_error_distribution(self):
        """Plot distribution of E8 lattice errors"""
        errors = [r['error'] for r in self.results]
        
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=100, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(np.percentile(errors, 5), color='red', linestyle='--', 
                   label='5th percentile (low error)')
        plt.xlabel('E8 Lattice Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of E8 Lattice Decoding Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f"Error statistics:")
        print(f"  Mean: {np.mean(errors):.4f}")
        print(f"  Std: {np.std(errors):.4f}")
        print(f"  Low error threshold (5%): {np.percentile(errors, 5):.4f}")
    
    def plot_byte_frequency(self):
        """Plot frequency of decoded bytes"""
        bytes_list = [r['byte'] for r in self.results]
        
        # Create frequency histogram
        freq, bins = np.histogram(bytes_list, bins=256, range=(0, 255))
        
        plt.figure(figsize=(12, 6))
        plt.bar(bins[:-1], freq, width=1, alpha=0.7, color='green', edgecolor='black')
        
        # Mark special bytes
        special_bytes = {0: 'NULL', 8: 'E8', 137: 'α⁻¹', 255: 'FF'}
        for byte_val, label in special_bytes.items():
            if byte_val < len(freq):
                plt.bar(byte_val, freq[byte_val], width=1, color='red', edgecolor='black')
                plt.text(byte_val, freq[byte_val], label, ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Byte Value')
        plt.ylabel('Frequency')
        plt.title('Frequency Distribution of Decoded Bytes')
        plt.xlim(0, 255)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_e8_vectors(self, num_vectors=1000):
        """Visualize E8 vectors in 2D projection"""
        vectors = [r['e8_vector'] for r in self.results[:num_vectors]]
        errors = [r['error'] for r in self.results[:num_vectors]]
        
        # Project to 2D for visualization
        projected = np.array(vectors)[:, :2]  # Use first two dimensions
        
        plt.figure(figsize=(10, 8))
        
        # Color by error (lower error = more red)
        norm_errors = np.array(errors) / max(errors)
        colors = hsv_to_rgb(np.stack([norm_errors, np.ones_like(norm_errors), 
                                      np.ones_like(norm_errors)], axis=1))
        
        plt.scatter(projected[:, 0], projected[:, 1], c=colors, alpha=0.6, 
                   edgecolors='black', linewidths=0.5)
        
        # Mark E8 lattice points
        lattice_points = []
        for x in range(-3, 4):
            for y in range(-3, 4):
                if (x + y) % 2 == 0:  # Even sum condition
                    lattice_points.append([x, y])
        
        lattice_points = np.array(lattice_points)
        plt.scatter(lattice_points[:, 0], lattice_points[:, 1], 
                   c='black', marker='+', s=100, label='E8 Lattice Points')
        
        plt.xlabel('E8 Dimension 1 (projected)')
        plt.ylabel('E8 Dimension 2 (projected)')
        plt.title('E8 Vector Projection (color = decoding error)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()
    
    def analyze_message_clusters(self):
        """Identify clusters of similar messages"""
        bytes_list = [r['byte'] for r in self.results]
        
        # Convert to binary representation
        binary_reps = []
        for byte_val in bytes_list:
            binary = format(byte_val, '08b')
            binary_reps.append([int(b) for b in binary])
        
        binary_matrix = np.array(binary_reps)
        
        # Compute Hamming distances between messages
        from scipy.spatial.distance import pdist, squareform
        distances = pdist(binary_matrix, metric='hamming')
        distance_matrix = squareform(distances)
        
        # Find clusters with small distances
        threshold = 0.25  # Max 2 bits different out of 8
        clusters = []
        visited = set()
        
        for i in range(len(binary_matrix)):
            if i not in visited:
                cluster = [i]
                for j in range(i+1, len(binary_matrix)):
                    if distance_matrix[i, j] < threshold:
                        cluster.append(j)
                        visited.add(j)
                
                if len(cluster) > 1:  # Only keep non-trivial clusters
                    clusters.append(cluster)
                visited.add(i)
        
        print(f"Found {len(clusters)} clusters of similar messages")
        
        for i, cluster in enumerate(clusters[:5]):  # Show first 5 clusters
            print(f"\nCluster {i+1} (size: {len(cluster)}):")
            cluster_bytes = [bytes_list[idx] for idx in cluster]
            print(f"  Bytes: {cluster_bytes}")
            print(f"  Hex: {[hex(b) for b in cluster_bytes]}")
            
            # Try to interpret as ASCII
            ascii_chars = ''.join([chr(b) if 32 <= b < 127 else '.' for b in cluster_bytes])
            print(f"  ASCII: {ascii_chars}")
        
        return clusters
    
    def visualize_spectral_gap(self):
        """Visualize the spectral gap concept from E8"""
        # Create synthetic E8 spectrum
        e8_roots = np.random.randn(240, 8)
        e8_norms = np.linalg.norm(e8_roots, axis=1)
        
        # Add minimal norm √2
        min_norm = np.sqrt(2)
        
        plt.figure(figsize=(10, 6))
        
        # Plot E8 root norms
        plt.hist(e8_norms, bins=50, alpha=0.7, label='E8 Root Distribution')
        plt.axvline(min_norm, color='red', linestyle='--', 
                   label=f'Minimal Norm = √2 ≈ {min_norm:.3f}')
        
        # Add spectral gap region
        plt.axvspan(min_norm, min_norm*1.5, alpha=0.2, color='green', 
                   label='Spectral Gap Region')
        
        plt.xlabel('Vector Norm')
        plt.ylabel('Frequency')
        plt.title('E8 Root Lattice Spectral Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Usage
if __name__ == "__main__":
    # Load and visualize results
    visualizer = E8Visualizer()
    
    print("Generating visualizations...")
    visualizer.plot_error_distribution()
    visualizer.plot_byte_frequency()
    visualizer.plot_e8_vectors(num_vectors=500)
    visualizer.analyze_message_clusters()
    visualizer.visualize_spectral_gap()
