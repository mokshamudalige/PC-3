"""
Comprehensive Performance Graph Generator for Gaussian Blur Parallel Implementations
Generates all 12 performance graphs in a single script.

Usage:
    python generate_performance_graphs.py

Requirements:
    pip install matplotlib numpy
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================================
# SECTION 1: CONFIGURE YOUR ACTUAL PERFORMANCE DATA HERE
# ============================================================================

# Serial baseline time (seconds)
SERIAL_TIME = 0.05000  # TODO: Replace with your actual serial execution time

# OpenMP Performance Data
openmp_threads = [1, 2, 4, 8, 16]
openmp_times = [0.05000, 0.02700, 0.01500, 0.01000, 0.00800]  # TODO: Replace with actual values

# MPI Performance Data
mpi_processes = [1, 2, 4, 8, 16]
mpi_times = [0.05500, 0.03200, 0.02000, 0.01500, 0.01300]  # TODO: Replace with actual values

# CUDA Performance Data
cuda_block_sizes = ['8×8', '16×16', '32×32']
cuda_basic_times = [0.001500, 0.001200, 0.001400]  # TODO: Replace with actual values
cuda_shared_times = [0.001000, 0.000800, 0.001100]  # TODO: Replace with actual values

# ============================================================================
# SECTION 2: CALCULATED METRICS (Auto-computed from above data)
# ============================================================================

# Calculate speedup and efficiency for OpenMP
openmp_speedup = [SERIAL_TIME / t for t in openmp_times]
openmp_efficiency = [(s / p) * 100 for s, p in zip(openmp_speedup, openmp_threads)]

# Calculate speedup and efficiency for MPI
mpi_speedup = [SERIAL_TIME / t for t in mpi_times]
mpi_efficiency = [(s / p) * 100 for s, p in zip(mpi_speedup, mpi_processes)]

# Calculate speedup for CUDA
cuda_basic_speedup = [SERIAL_TIME / t for t in cuda_basic_times]
cuda_shared_speedup = [SERIAL_TIME / t for t in cuda_shared_times]

# Ideal linear speedup for comparison
ideal_speedup = openmp_threads  # Same for MPI

# ============================================================================
# SECTION 3: GRAPH STYLING CONFIGURATION
# ============================================================================

# Set publication-quality defaults
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 8

# Color scheme
COLOR_OPENMP = '#2E86AB'      # Blue
COLOR_MPI = '#A23B72'         # Purple
COLOR_CUDA_BASIC = '#F18F01'  # Orange
COLOR_CUDA_SHARED = '#C73E1D' # Red
COLOR_IDEAL = '#6C757D'       # Gray
COLOR_GRID = '#E0E0E0'        # Light gray

# Create output directory
OUTPUT_DIR = 'performance_graphs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# SECTION 4: GRAPH GENERATION FUNCTIONS
# ============================================================================

def save_figure(filename):
    """Save figure with high resolution and tight layout."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {filename}")
    plt.close()


def graph_01_openmp_time():
    """Figure 1: OpenMP Threads vs. Execution Time"""
    plt.figure()
    plt.plot(openmp_threads, openmp_times, 'o-', color=COLOR_OPENMP, label='OpenMP')
    plt.xlabel('Number of Threads')
    plt.ylabel('Execution Time (seconds)')
    plt.title('OpenMP: Threads vs. Execution Time')
    plt.grid(True, alpha=0.3, color=COLOR_GRID)
    plt.legend()
    plt.xticks(openmp_threads)
    save_figure('figure_01_openmp_execution_time.png')


def graph_02_openmp_speedup():
    """Figure 2: OpenMP Threads vs. Speedup"""
    plt.figure()
    plt.plot(openmp_threads, openmp_speedup, 'o-', color=COLOR_OPENMP, label='Actual Speedup')
    plt.plot(openmp_threads, ideal_speedup, '--', color=COLOR_IDEAL, label='Ideal Linear Speedup')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.title('OpenMP: Threads vs. Speedup')
    plt.grid(True, alpha=0.3, color=COLOR_GRID)
    plt.legend()
    plt.xticks(openmp_threads)
    save_figure('figure_02_openmp_speedup.png')


def graph_03_openmp_efficiency():
    """Figure 3: OpenMP Threads vs. Efficiency"""
    plt.figure()
    plt.plot(openmp_threads, openmp_efficiency, 'o-', color=COLOR_OPENMP)
    plt.xlabel('Number of Threads')
    plt.ylabel('Parallel Efficiency (%)')
    plt.title('OpenMP: Threads vs. Parallel Efficiency')
    plt.grid(True, alpha=0.3, color=COLOR_GRID)
    plt.xticks(openmp_threads)
    plt.ylim(0, 110)
    save_figure('figure_03_openmp_efficiency.png')


def graph_04_mpi_time():
    """Figure 4: MPI Processes vs. Execution Time"""
    plt.figure()
    plt.plot(mpi_processes, mpi_times, 's-', color=COLOR_MPI, label='MPI')
    plt.xlabel('Number of Processes')
    plt.ylabel('Execution Time (seconds)')
    plt.title('MPI: Processes vs. Execution Time')
    plt.grid(True, alpha=0.3, color=COLOR_GRID)
    plt.legend()
    plt.xticks(mpi_processes)
    save_figure('figure_04_mpi_execution_time.png')


def graph_05_mpi_speedup():
    """Figure 5: MPI Processes vs. Speedup"""
    plt.figure()
    plt.plot(mpi_processes, mpi_speedup, 's-', color=COLOR_MPI, label='Actual Speedup')
    plt.plot(mpi_processes, ideal_speedup, '--', color=COLOR_IDEAL, label='Ideal Linear Speedup')
    plt.xlabel('Number of Processes')
    plt.ylabel('Speedup')
    plt.title('MPI: Processes vs. Speedup')
    plt.grid(True, alpha=0.3, color=COLOR_GRID)
    plt.legend()
    plt.xticks(mpi_processes)
    save_figure('figure_05_mpi_speedup.png')


def graph_06_mpi_efficiency():
    """Figure 6: MPI Processes vs. Efficiency"""
    plt.figure()
    plt.plot(mpi_processes, mpi_efficiency, 's-', color=COLOR_MPI)
    plt.xlabel('Number of Processes')
    plt.ylabel('Parallel Efficiency (%)')
    plt.title('MPI: Processes vs. Parallel Efficiency')
    plt.grid(True, axis='both', alpha=0.3, color=COLOR_GRID)
    plt.xticks(mpi_processes)
    plt.ylim(0, 110)
    save_figure('figure_06_mpi_efficiency.png')


def graph_07_cuda_time():
    """Figure 7: CUDA Block Size vs. Execution Time"""
    x = np.arange(len(cuda_block_sizes))
    width = 0.35
    
    plt.figure()
    plt.bar(x - width/2, cuda_basic_times, width, label='Basic Kernel', color=COLOR_CUDA_BASIC)
    plt.bar(x + width/2, cuda_shared_times, width, label='Shared Memory Kernel', color=COLOR_CUDA_SHARED)
    
    plt.xlabel('Block Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('CUDA: Block Size vs. Execution Time')
    plt.xticks(x, cuda_block_sizes)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3, color=COLOR_GRID)
    save_figure('figure_07_cuda_execution_time.png')


def graph_08_cuda_speedup():
    """Figure 8: CUDA Block Size vs. Speedup"""
    x = np.arange(len(cuda_block_sizes))
    width = 0.35
    
    plt.figure()
    plt.bar(x - width/2, cuda_basic_speedup, width, label='Basic Kernel', color=COLOR_CUDA_BASIC)
    plt.bar(x + width/2, cuda_shared_speedup, width, label='Shared Memory Kernel', color=COLOR_CUDA_SHARED)
    
    plt.xlabel('Block Size')
    plt.ylabel('Speedup vs. Serial')
    plt.title('CUDA: Block Size vs. Speedup')
    plt.xticks(x, cuda_block_sizes)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3, color=COLOR_GRID)
    save_figure('figure_08_cuda_speedup.png')


def graph_09_cuda_comparison():
    """Figure 9: CUDA Kernel Comparison (Basic vs. Shared Memory)"""
    categories = cuda_block_sizes
    
    # Calculate percentage improvement
    improvements = [(1 - shared/basic) * 100 for basic, shared in zip(cuda_basic_times, cuda_shared_times)]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(categories))
    
    # Create grouped bar chart
    width = 0.35
    plt.bar(x - width/2, cuda_basic_times, width, label='Basic Kernel', color=COLOR_CUDA_BASIC)
    plt.bar(x + width/2, cuda_shared_times, width, label='Shared Memory', color=COLOR_CUDA_SHARED)
    
    # Add percentage labels
    for i, improvement in enumerate(improvements):
        plt.text(i, max(cuda_basic_times[i], cuda_shared_times[i]) * 1.1, 
                f'{improvement:.1f}% faster', ha='center', fontsize=10, fontweight='bold')
    
    plt.xlabel('Block Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('CUDA: Basic vs. Shared Memory Kernel Performance')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3, color=COLOR_GRID)
    save_figure('figure_09_cuda_kernel_comparison.png')


def graph_10_comparative_time():
    """Figure 10: Comparative Execution Time (All Implementations)"""
    # Use optimal configurations for comparison
    implementations = ['Serial', 'OpenMP\n(16 threads)', 'MPI\n(16 processes)', 
                      'CUDA Basic\n(16×16)', 'CUDA Shared\n(16×16)']
    times = [SERIAL_TIME, openmp_times[-1], mpi_times[-1], cuda_basic_times[1], cuda_shared_times[1]]
    colors = [COLOR_IDEAL, COLOR_OPENMP, COLOR_MPI, COLOR_CUDA_BASIC, COLOR_CUDA_SHARED]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(implementations, times, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{time*1000:.2f} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylabel('Execution Time (seconds)')
    plt.title('Comparative Execution Time Across All Implementations')
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(True, axis='y', alpha=0.3, color=COLOR_GRID, which='both')
    save_figure('figure_10_comparative_execution_time.png')


def graph_11_comparative_speedup():
    """Figure 11: Comparative Speedup Analysis"""
    implementations = ['OpenMP\n(16 threads)', 'MPI\n(16 processes)', 
                      'CUDA Basic\n(16×16)', 'CUDA Shared\n(16×16)']
    speedups = [openmp_speedup[-1], mpi_speedup[-1], cuda_basic_speedup[1], cuda_shared_speedup[1]]
    colors = [COLOR_OPENMP, COLOR_MPI, COLOR_CUDA_BASIC, COLOR_CUDA_SHARED]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(implementations, speedups, color=colors, alpha=0.8)
    
    # Add value labels
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.1f}×', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylabel('Speedup vs. Serial Baseline')
    plt.title('Comparative Speedup Analysis (Best Configuration)')
    plt.grid(True, axis='y', alpha=0.3, color=COLOR_GRID)
    save_figure('figure_11_comparative_speedup.png')


def graph_12_efficiency_comparison():
    """Figure 12: Parallel Efficiency Comparison"""
    plt.figure(figsize=(10, 6))
    
    # Plot OpenMP efficiency
    plt.plot(openmp_threads, openmp_efficiency, 'o-', color=COLOR_OPENMP, 
            label='OpenMP', linewidth=2.5, markersize=8)
    
    # Plot MPI efficiency
    plt.plot(mpi_processes, mpi_efficiency, 's-', color=COLOR_MPI, 
            label='MPI', linewidth=2.5, markersize=8)
    
    # Plot 100% efficiency reference line
    plt.axhline(y=100, color=COLOR_IDEAL, linestyle='--', linewidth=1.5, label='Ideal (100%)')
    
    plt.xlabel('Number of Processing Units (Threads/Processes)')
    plt.ylabel('Parallel Efficiency (%)')
    plt.title('Parallel Efficiency: OpenMP vs. MPI')
    plt.legend()
    plt.grid(True, alpha=0.3, color=COLOR_GRID)
    plt.xticks(openmp_threads)
    plt.ylim(0, 110)
    save_figure('figure_12_efficiency_comparison.png')


# ============================================================================
# SECTION 5: MAIN EXECUTION
# ============================================================================

def main():
    """Generate all performance graphs."""
    print("\n" + "="*70)
    print("  Gaussian Blur Performance Graph Generator")
    print("="*70 + "\n")
    
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}\n")
    print("Generating graphs...\n")
    
    # OpenMP graphs (3 graphs)
    print("OpenMP Graphs:")
    graph_01_openmp_time()
    graph_02_openmp_speedup()
    graph_03_openmp_efficiency()
    
    # MPI graphs (3 graphs)
    print("\nMPI Graphs:")
    graph_04_mpi_time()
    graph_05_mpi_speedup()
    graph_06_mpi_efficiency()
    
    # CUDA graphs (3 graphs)
    print("\nCUDA Graphs:")
    graph_07_cuda_time()
    graph_08_cuda_speedup()
    graph_09_cuda_comparison()
    
    # Comparative graphs (3 graphs)
    print("\nComparative Analysis Graphs:")
    graph_10_comparative_time()
    graph_11_comparative_speedup()
    graph_12_efficiency_comparison()
    
    print("\n" + "="*70)
    print(f"✓ Successfully generated 12 graphs in '{OUTPUT_DIR}/' directory")
    print("="*70 + "\n")
    
    print("Next steps:")
    print("1. Review the generated graphs in the performance_graphs/ folder")
    print("2. Update Appendix A in your report with these figure references")
    print("3. Insert the PNG files into your report document")
    print()


if __name__ == "__main__":
    main()
