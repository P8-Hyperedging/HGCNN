"""
Module for generating diagnostic statistics and visualizations for quality score distributions.
Data is collected during training and GIF is generated directly without intermediate PNG files.
"""

import os
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import numpy as np
import io
import time


class QualityStatistics:
    """Collect and manage quality score diagnostics without generating plots during training."""
    
    # Class-level storage for diagnostic data
    diagnostic_data = []
    frame_skip = 1  # Will be set during training to match GIF rendering
    
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
        
        # Only print statistics for frames that will be rendered in GIF (based on frame_skip)
        if forward_pass_num % QualityStatistics.frame_skip == 0:
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
    def figure_to_pil_image(fig):
        """
        Convert a matplotlib figure to a PIL Image without saving to disk.
        
        Args:
            fig: matplotlib Figure object
        
        Returns:
            PIL Image object
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        img.load()  # Force load before closing buffer
        # Convert to RGB if necessary (in case of RGBA)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    
    @staticmethod
    def create_diagnostics_gif_direct(stats_dir, frame_skip=1):
        """
        Create a GIF directly from collected diagnostic data without saving intermediate PNGs.
        All figures are rendered to PIL Images in memory.
        
        Args:
            stats_dir: Directory to save the final GIF
            frame_skip: Only render every Nth frame (e.g., frame_skip=2 renders frames 0, 2, 4, ...)
        
        Returns:
            Path to the created GIF file, or None if no data collected
        """
        if not QualityStatistics.diagnostic_data:
            print("No diagnostic data to create GIF from.")
            return None
        
        # Get frames to render based on frame_skip
        frame_indices = range(0, len(QualityStatistics.diagnostic_data), frame_skip)
        total_frames = len(QualityStatistics.diagnostic_data)
        frames_to_render = len(frame_indices)
        
        print(f"\n{'='*60}")
        print(f"GIF Generation Starting")
        print(f"Total frames: {total_frames} | Rendering: {frames_to_render} (skip={frame_skip})")
        print(f"{'='*60}")
        
        start_time = time.time()
        images = []
        
        for render_idx, forward_pass_num in enumerate(frame_indices):
            frame_start = time.time()
            
            data = QualityStatistics.diagnostic_data[forward_pass_num]
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
            
            # Convert figure to PIL Image directly (no disk I/O)
            pil_image = QualityStatistics.figure_to_pil_image(fig)
            images.append(pil_image)
            
            plt.close(fig)
            
            frame_time = time.time() - frame_start
            progress = (render_idx + 1) / frames_to_render * 100
            print(f"[{render_idx + 1:>3d}/{frames_to_render}] Frame {forward_pass_num:>3d} rendered ({progress:>5.1f}%) - {frame_time:.2f}s")
        
        render_time = time.time() - start_time
        print(f"\nAll {total_frames} frames rendered in {render_time:.2f}s")
        
        # Create GIF from PIL Images
        print("\nAssembling GIF...", end='', flush=True)
        os.makedirs(stats_dir, exist_ok=True)
        gif_path = os.path.join(stats_dir, 'quality_diagnostics.gif')
        
        if images:
            gif_start = time.time()
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=100,
                loop=0
            )
            gif_time = time.time() - gif_start
            print(f" Done! ({gif_time:.2f}s)")
            
            total_time = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"GIF created successfully!")
            print(f"File: {gif_path}")
            print(f"Total time: {total_time:.2f}s")
            print(f"{'='*60}")
        else:
            print("No images to create GIF from.")
            return None
        
        return gif_path
    
    @staticmethod
    def finalize_diagnostics(stats_dir, frame_skip=1):
        """
        Generate GIF directly from collected diagnostic data after training completes.
        No intermediate PNG files are created - everything happens in memory.
        
        Args:
            stats_dir: Directory to save outputs
            frame_skip: Only render every Nth frame (e.g., frame_skip=2 renders frames 0, 2, 4, ...)
        
        Returns:
            Path to the created GIF file, or None if no data collected
        """
        if not QualityStatistics.diagnostic_data:
            print("No diagnostic data to finalize.")
            return None
        
        return QualityStatistics.create_diagnostics_gif_direct(stats_dir, frame_skip=frame_skip)
