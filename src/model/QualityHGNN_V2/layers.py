import math
import os
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from PIL import Image
import glob


class QHGNN_conv_v2(nn.Module):
    call_count = 0  # Class variable to track forward passes
    
    def __init__(self, in_ft, out_ft, quality=False, bias=True):
        super(QHGNN_conv_v2, self).__init__()

        self.quality = quality
        self.weight = Parameter(torch.Tensor(in_ft, out_ft)) # Create new feature matrix for hidden layer
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, LS: torch.Tensor, Q: torch.Tensor, RS: torch.Tensor):
        """Q is a 1D vector of diagonal values (not a full matrix)."""
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        if not self.quality:
            G = LS.matmul(RS)
            x = G.matmul(x)
            return x
        with torch.no_grad():
            membership = (LS > 0).float()                          # (N, E)
            nodes_per_edge = membership.sum(dim=0).clamp(min=1)    # (E,)

            # Centroids for all hyperedges at once: (E, F)
            centroids = membership.T.matmul(x) / nodes_per_edge.unsqueeze(1)

            # Compute total distance per hyperedge in chunks to limit memory
            E = LS.shape[1]
            chunk_size = 100
            total_dists = torch.zeros(E, device=x.device)

            for start in range(0, E, chunk_size):
                end = min(start + chunk_size, E)
                # (N, chunk, F) - broadcast node features against chunk centroids
                diffs = x.unsqueeze(1) - centroids[start:end].unsqueeze(0)
                dists = diffs.norm(dim=2)                          # (N, chunk)
                dists = dists * membership[:, start:end]           # zero out non-members
                total_dists[start:end] = dists.sum(dim=0)

            # Normalize total_dists by its mean to keep distance_scores in reasonable range
            mean_total_dists = total_dists.mean()
            total_dists_normalized = total_dists / (mean_total_dists + 1e-8)  # Add small epsilon to avoid division by zero
            distance_scores = 1.0 / (1.0 + total_dists_normalized)           # (E,)
            Q_updated = torch.clamp(distance_scores * Q, min=0, max=10)

            # ========== DIAGNOSTICS ==========
            stats_dir = os.path.join(os.path.dirname(__file__), '..', 'statistics')
            os.makedirs(stats_dir, exist_ok=True)
            
            # Log initial Q and ranges
            Q_cpu = Q.cpu().detach().numpy()
            total_dists_cpu = total_dists.cpu().detach().numpy()
            total_dists_normalized_cpu = total_dists_normalized.cpu().detach().numpy()
            distance_scores_cpu = distance_scores.cpu().detach().numpy()
            Q_updated_cpu = Q_updated.cpu().detach().numpy()
            
            print(f"\n=== Forward Pass {QHGNN_conv_v2.call_count} ===")
            print(f"Initial Q - Min: {Q_cpu.min():.6f}, Max: {Q_cpu.max():.6f}, Mean: {Q_cpu.mean():.6f}")
            print(f"Total Dists (raw) - Min: {total_dists_cpu.min():.6f}, Max: {total_dists_cpu.max():.6f}, Mean: {total_dists_cpu.mean():.6f}")
            print(f"Total Dists (normalized) - Min: {total_dists_normalized_cpu.min():.6f}, Max: {total_dists_normalized_cpu.max():.6f}, Mean: {total_dists_normalized_cpu.mean():.6f}")
            print(f"Distance Scores - Min: {distance_scores_cpu.min():.6f}, Max: {distance_scores_cpu.max():.6f}, Mean: {distance_scores_cpu.mean():.6f}")
            print(f"Q Updated - Min: {Q_updated_cpu.min():.6f}, Max: {Q_updated_cpu.max():.6f}, Mean: {Q_updated_cpu.mean():.6f}")
            print(f"Num Q values at exactly 0.0: {(Q_updated_cpu == 0.0).sum()} / {len(Q_updated_cpu)}")
            
            # Create figure with 4 subplots
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Forward Pass {QHGNN_conv_v2.call_count}', fontsize=16, fontweight='bold', y=0.995)
            
            # Plot 1: Quality Score Distribution
            axes[0, 0].hist(Q_updated_cpu, bins=50, edgecolor='black', alpha=0.7, color='blue')
            axes[0, 0].set_xlabel('Quality Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title(f'Quality Score Distribution (n={len(Q_updated_cpu)})')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Distance Scores Distribution
            axes[0, 1].hist(distance_scores_cpu, bins=50, edgecolor='black', alpha=0.7, color='green')
            axes[0, 1].set_xlabel('Distance Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title(f'Distance Scores Distribution (min={distance_scores_cpu.min():.4f}, max={distance_scores_cpu.max():.4f})')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Total Distances Distribution (normalized) - with log scale
            axes[1, 0].hist(total_dists_normalized_cpu, bins=50, edgecolor='black', alpha=0.7, color='orange')
            axes[1, 0].set_xlabel('Total Distance (Normalized by Mean)')
            axes[1, 0].set_ylabel('Frequency (log scale)')
            axes[1, 0].set_yscale('log')
            axes[1, 0].set_title(f'Normalized Total Distances (min={total_dists_normalized_cpu.min():.4f}, max={total_dists_normalized_cpu.max():.4f})')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Scatter plot of distance_scores vs Q_updated
            scatter = axes[1, 1].scatter(distance_scores_cpu, Q_updated_cpu, alpha=0.3, s=1, c=Q_cpu, cmap='viridis')
            axes[1, 1].set_xlabel('Distance Score')
            axes[1, 1].set_ylabel('Q Updated')
            axes[1, 1].set_title('Relationship: Distance Scores vs Q Updated (colored by Initial Q)')
            axes[1, 1].grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=axes[1, 1])
            cbar.set_label('Initial Q')
            
            plt.tight_layout()
            plot_path = os.path.join(stats_dir, f'diagnostics_{QHGNN_conv_v2.call_count}.png')
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            QHGNN_conv_v2.call_count += 1
            G = (LS * Q_updated.unsqueeze(0)).matmul(RS)

        x = G.matmul(x)

        return x

    @staticmethod
    def create_diagnostics_gif(stats_dir=None):
        """
        Create an animated GIF from all diagnostic PNG files in the statistics folder.
        Deletes PNG files after GIF is created.
        
        Args:
            stats_dir: Path to statistics folder. If None, uses default path relative to this file.
        
        Returns:
            Path to the created GIF file, or None if no images found.
        """
        if stats_dir is None:
            stats_dir = os.path.join(os.path.dirname(__file__), '..', 'statistics')
        
        stats_dir = os.path.abspath(stats_dir)
        
        # Find all diagnostic PNG files and sort by number
        png_files = sorted(glob.glob(os.path.join(stats_dir, 'diagnostics_*.png')),
                          key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        
        if not png_files:
            print(f"No diagnostic PNG files found in {stats_dir}")
            return None
        
        print(f"Creating GIF from {len(png_files)} diagnostic images...")
        
        # Load images
        images = []
        for png_file in png_files:
            img = Image.open(png_file)
            images.append(img)
        
        # Create GIF (1000 ms = 1 second per frame)
        gif_path = os.path.join(stats_dir, 'quality_diagnostics.gif')
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=1000,  # 1 second per frame
            loop=0  # Loop infinitely
        )
        
        print(f"GIF created successfully: {gif_path}")
        
        # Delete PNG files after GIF creation
        print(f"Deleting {len(png_files)} PNG diagnostic files...")
        for png_file in png_files:
            try:
                os.remove(png_file)
            except Exception as e:
                print(f"Warning: Could not delete {png_file}: {e}")
        
        print("PNG files deleted. Only GIF remains.")
        return gif_path
