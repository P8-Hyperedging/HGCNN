"""
Module for generating diagnostic statistics and visualizations for quality score distributions.
Data is collected during training and plots are generated after training completes.
"""

import os
import glob
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


class QualityStatistics:
    """Collect and manage quality score diagnostics without generating plots during training."""
    
    # Class-level storage for diagnostic data
    diagnostic_data = []
    
    @staticmethod
    def collect_diagnostic_data(Q, total_dists, total_dists_normalized, distance_scores, Q_updated):
        """
        Collect diagnostic data during forward passes instead of generating plots.
        This significantly reduces VRAM usage.
        
        Args:
            Q: Initial quality scores tensor
            total_dists: Raw total distances tensor
            total_dists_normalized: Normalized total distances tensor
            distance_scores: Distance scores tensor
            Q_updated: Updated quality scores tensor
        """
        # Convert to CPU and numpy (minimal memory overhead)
        data = {
            'Q': Q.cpu().detach().numpy(),
            'total_dists': total_dists.cpu().detach().numpy(),
            'total_dists_normalized': total_dists_normalized.cpu().detach().numpy(),
            'distance_scores': distance_scores.cpu().detach().numpy(),
            'Q_updated': Q_updated.cpu().detach().numpy(),
        }
        
        QualityStatistics.diagnostic_data.append(data)
        forward_pass_num = len(QualityStatistics.diagnostic_data) - 1
        
        # Log statistics to console
        Q_cpu = data['Q']
        total_dists_cpu = data['total_dists']
        total_dists_normalized_cpu = data['total_dists_normalized']
        distance_scores_cpu = data['distance_scores']
        Q_updated_cpu = data['Q_updated']
        
        print(f"\n=== Forward Pass {forward_pass_num} ===")
        print(f"Initial Q - Min: {Q_cpu.min():.6f}, Max: {Q_cpu.max():.6f}, Mean: {Q_cpu.mean():.6f}")
        print(f"Total Dists (raw) - Min: {total_dists_cpu.min():.6f}, Max: {total_dists_cpu.max():.6f}, Mean: {total_dists_cpu.mean():.6f}")
        print(f"Total Dists (normalized) - Min: {total_dists_normalized_cpu.min():.6f}, Max: {total_dists_normalized_cpu.max():.6f}, Mean: {total_dists_normalized_cpu.mean():.6f}")
        print(f"Distance Scores - Min: {distance_scores_cpu.min():.6f}, Max: {distance_scores_cpu.max():.6f}, Mean: {distance_scores_cpu.mean():.6f}")
        print(f"Q Updated - Min: {Q_updated_cpu.min():.6f}, Max: {Q_updated_cpu.max():.6f}, Mean: {Q_updated_cpu.mean():.6f}")
        print(f"Num Q values at exactly 0.0: {(Q_updated_cpu == 0.0).sum()} / {len(Q_updated_cpu)}")
    
    @staticmethod
    def generate_diagnostic_plots(stats_dir):
        """
        Generate all diagnostic PNG plots from collected data after training completes.
        
        Args:
            stats_dir: Directory to save the plots
        
        Returns:
            List of paths to generated PNG files
        """
        if not QualityStatistics.diagnostic_data:
            print("No diagnostic data collected to generate plots.")
            return []
        
        os.makedirs(stats_dir, exist_ok=True)
        png_paths = []
        
        print(f"\nGenerating {len(QualityStatistics.diagnostic_data)} diagnostic plots...")
        
        for forward_pass_num, data in enumerate(QualityStatistics.diagnostic_data):
            Q_cpu = data['Q']
            total_dists_normalized_cpu = data['total_dists_normalized']
            distance_scores_cpu = data['distance_scores']
            Q_updated_cpu = data['Q_updated']
            
            # Create figure with 4 subplots
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Forward Pass {forward_pass_num}', fontsize=16, fontweight='bold', y=0.995)
            
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
            plot_path = os.path.join(stats_dir, f'diagnostics_{forward_pass_num}.png')
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            png_paths.append(plot_path)
        
        print(f"Generated {len(png_paths)} diagnostic plots.")
        return png_paths
    
    @staticmethod
    def finalize_diagnostics(stats_dir):
        """
        Generate all diagnostic plots and GIF after training completes.
        This should be called once after training finishes.
        
        Args:
            stats_dir: Directory to save outputs
        
        Returns:
            Path to the created GIF file, or None if no data collected
        """
        if not QualityStatistics.diagnostic_data:
            print("No diagnostic data to finalize.")
            return None
        
        # Generate all PNG plots from collected data
        QualityStatistics.generate_diagnostic_plots(stats_dir)
        
        # Create GIF from the generated PNGs
        QualityStatistics.create_diagnostics_gif(stats_dir)
        
        return os.path.join(stats_dir, 'quality_diagnostics.gif')
    
    @staticmethod
    def create_diagnostics_gif(stats_dir):
        """
        Create an animated GIF from all diagnostic PNG files in the statistics folder.
        Deletes PNG files after GIF is created.
        
        Args:
            stats_dir: Path to statistics folder
        
        Returns:
            Path to the created GIF file, or None if no images found.
        """
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
