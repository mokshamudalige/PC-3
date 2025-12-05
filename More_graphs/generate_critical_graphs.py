"""
Critical Performance Graph Generator - Top 4 Essential Graphs
Generates only the 4 most important graphs for the formal report.

Usage:
    python generate_critical_graphs.py

Requirements:
    pip install matplotlib numpy
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================================
# CONFIGURE YOUR ACTUAL PERFORMANCE DATA HERE
# ============================================================================

# Serial baseline time (seconds) - Median of [0.038, 0.061, 0.023, 0.036, 0.029]
SERIAL_TIME = 0.036  # Median of 5 runs

# OpenMP Performance Data (using parallel execution times)
# OMP_NUM_THREADS: 1, 2, 4, 8 (note: 16 exceeded available cores)
openmp_threads = [1, 2, 4, 8]
openmp_times = [0.016496, 0.013976, 0.011291, 0.005007]  # Parallel execution times

# MPI Performance Data (np = 8 and 16 failed due to insufficient cores)
mpi_processes = [1, 2, 4]
mpi_times = [0.001402, 0.003981, 0.006535]  # Execution times

# CUDA Performance Data (16x16 block size only, basic vs shared)
# Basic kernel median: [0.000085, 0.000084, 0.000083, 0.000083, 0.000087] = 0.000084
# Shared kernel median: [0.000111, 0.000113, 0.000113, 0.000116, 0.000110] = 0.000113
cuda_block_sizes = ['16×16']  # Only tested 16x16
cuda_basic_times = [0.000084]  # Median of 5 runs
cuda_shared_times = [0.000113]  # Median of 5 runs

# ============================================================================
# CALCULATED METRICS
# ============================================================================

openmp_speedup = [SERIAL_TIME / t for t in openmp_times]
openmp_efficiency = [(s / p) * 100 for s, p in zip(openmp_speedup, openmp_threads)]

mpi_speedup = [SERIAL_TIME / t for t in mpi_times]
mpi_efficiency = [(s / p) * 100 for s, p in zip(mpi_speedup, mpi_processes)]

cuda_basic_speedup = [SERIAL_TIME / t for t in cuda_basic_times]
cuda_shared_speedup = [SERIAL_TIME / t for t in cuda_shared_times]

ideal_speedup = openmp_threads

# ============================================================================
# STYLING
# ============================================================================

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 8

COLOR_OPENMP = '#2E86AB'
COLOR_MPI = '#A23B72'
COLOR_CUDA_BASIC = '#F18F01'
COLOR_CUDA_SHARED = '#C73E1D'
COLOR_IDEAL = '#6C757D'
COLOR_GRID = '#E0E0E0'

OUTPUT_DIR = 'critical_graphs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_figure(filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ {filename}")
    plt.close()

# ============================================================================
# THE 4 CRITICAL GRAPHS
# ============================================================================

def graph_1_openmp_speedup_with_ideal():
    """CRITICAL GRAPH 1: OpenMP Speedup vs. Ideal (Amdahl's Law)"""
    plt.figure()
    plt.plot(openmp_threads, openmp_speedup, 'o-', color=COLOR_OPENMP, 
             label='Actual Speedup', linewidth=3, markersize=10)
    plt.plot(openmp_threads, ideal_speedup, '--', color=COLOR_IDEAL, 
             label='Ideal Linear Speedup', linewidth=2.5)
    
    # Add speedup values on points
    for x, y in zip(openmp_threads, openmp_speedup):
        plt.text(x, y + 0.15, f'{y:.2f}×', ha='center', fontsize=10, fontweight='bold')
    
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup vs. Serial Baseline')
    plt.title('OpenMP Speedup Analysis (Amdahl\'s Law Demonstration)')
    plt.grid(True, alpha=0.3, color=COLOR_GRID)
    plt.legend(loc='upper left')
    plt.xticks(openmp_threads)
    plt.xlim(0, 9)
    plt.ylim(0, max(max(openmp_speedup), max(ideal_speedup)) * 1.2)
    save_figure('critical_01_openmp_speedup_vs_ideal.png')


def graph_2_cuda_kernel_comparison():
    """CRITICAL GRAPH 2: CUDA Kernel Optimization Impact"""
    # Note: Shared memory was unexpectedly SLOWER for this workload
    improvements = [(1 - shared/basic) * 100 
                   for basic, shared in zip(cuda_basic_times, cuda_shared_times)]
    
    plt.figure(figsize=(8, 6))
    x = np.arange(len(cuda_block_sizes))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, cuda_basic_times, width, 
                    label='Basic Kernel (Global Memory)', color=COLOR_CUDA_BASIC)
    bars2 = plt.bar(x + width/2, cuda_shared_times, width, 
                    label='Shared Memory Kernel', color=COLOR_CUDA_SHARED)
    
    # Add percentage improvement labels (can be negative = slower)
    for i, improvement in enumerate(improvements):
        y_pos = max(cuda_basic_times[i], cuda_shared_times[i]) * 1.15
        
        # Use red for slowdown, wheat for speedup
        if improvement < 0:
            label_text = f'{abs(improvement):.1f}% SLOWER'
            label_color = '#ff6b6b'
        else:
            label_text = f'{improvement:.1f}% faster'
            label_color = 'wheat'
            
        plt.text(i, y_pos, label_text, 
                ha='center', fontsize=11, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor=label_color, alpha=0.5))
    
    # Add time labels on bars (in milliseconds for clarity)
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{height*1000:.3f}ms', ha='center', va='center', 
                fontsize=9, color='white', fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{height*1000:.3f}ms', ha='center', va='center', 
                fontsize=9, color='white', fontweight='bold')
    
    plt.xlabel('Block Size Configuration')
    plt.ylabel('Execution Time (seconds)')
    plt.title('CUDA Memory Optimization Impact: Basic vs. Shared Memory (16×16 blocks)')
    plt.xticks(x, cuda_block_sizes)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3, color=COLOR_GRID)
    save_figure('critical_02_cuda_kernel_comparison.png')


def graph_3_efficiency_comparison():
    """CRITICAL GRAPH 3: Parallel Efficiency Comparison"""
    plt.figure(figsize=(11, 6))
    
    plt.plot(openmp_threads, openmp_efficiency, 'o-', color=COLOR_OPENMP, 
            label='OpenMP', linewidth=3, markersize=10)
    plt.plot(mpi_processes, mpi_efficiency, 's-', color=COLOR_MPI, 
            label='MPI', linewidth=3, markersize=10)
    plt.axhline(y=100, color=COLOR_IDEAL, linestyle='--', 
               linewidth=2, label='Ideal (100%)')
    
    # Add efficiency percentages at key points (adjusted for actual data: 4 OpenMP + 3 MPI)
    for x, y in zip(openmp_threads, openmp_efficiency):
        plt.text(x, y + 3, f'{y:.1f}%', ha='center', fontsize=9, 
                color=COLOR_OPENMP, fontweight='bold')
    for x, y in zip(mpi_processes, mpi_efficiency):
        plt.text(x, y - 5, f'{y:.1f}%', ha='center', fontsize=9, 
                color=COLOR_MPI, fontweight='bold')
    
    plt.xlabel('Number of Processing Units (Threads/Processes)')
    plt.ylabel('Parallel Efficiency (%)')
    plt.title('Scalability Analysis: Parallel Efficiency Degradation')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3, color=COLOR_GRID)
    # Use combined x-axis showing all tested configurations
    all_x = sorted(set(openmp_threads + mpi_processes))
    plt.xticks(all_x)
    plt.xlim(0, max(all_x) + 1)
    plt.ylim(0, max(max(openmp_efficiency), max(mpi_efficiency)) + 10)
    save_figure('critical_03_efficiency_comparison.png')


def graph_4_comparative_speedup():
    """CRITICAL GRAPH 4: Comparative Speedup - All Implementations"""
    implementations = ['OpenMP\n(8 threads)', 'MPI\n(4 processes)', 
                      'CUDA Basic\n(16×16)', 'CUDA Shared\n(16×16)']
    speedups = [openmp_speedup[-1], mpi_speedup[-1], 
               cuda_basic_speedup[0], cuda_shared_speedup[0]]
    colors = [COLOR_OPENMP, COLOR_MPI, COLOR_CUDA_BASIC, COLOR_CUDA_SHARED]
    
    plt.figure(figsize=(11, 7))
    bars = plt.bar(implementations, speedups, color=colors, alpha=0.85, 
                   edgecolor='black', linewidth=1.5)
    
    # Add speedup values on top of bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{speedup:.1f}×\nspeedup', ha='center', va='bottom', 
                fontsize=13, fontweight='bold')
    
    # Add a reference line at 1× (serial baseline)
    plt.axhline(y=1, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
    plt.text(0.5, 1.5, 'Serial Baseline', fontsize=10, style='italic')
    
    plt.ylabel('Speedup vs. Serial Baseline', fontsize=14, fontweight='bold')
    plt.title('Comparative Performance: Best Configuration Speedup', 
             fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, axis='y', alpha=0.3, color=COLOR_GRID)
    plt.ylim(0, max(speedups) * 1.15)
    
    # Add subtitle with key insight about unexpected shared memory result
    plt.text(0.5, 0.98, 
            f'Note: CUDA Shared Memory was unexpectedly slower than Basic kernel for this workload',
            transform=plt.gca().transAxes, ha='center', va='top',
            fontsize=11, style='italic', bbox=dict(boxstyle='round', 
            facecolor='#ffcccc', alpha=0.5))
    
    save_figure('critical_04_comparative_speedup.png')


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("  CRITICAL PERFORMANCE GRAPHS GENERATOR")
    print("  Generating 4 Essential Graphs for Formal Report")
    print("="*70 + "\n")
    
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}/\n")
    
    print("Generating critical graphs...\n")
    
    print("1. OpenMP Speedup vs. Ideal (Amdahl's Law)")
    graph_1_openmp_speedup_with_ideal()
    
    print("2. CUDA Kernel Comparison (Memory Optimization)")
    graph_2_cuda_kernel_comparison()
    
    print("3. Efficiency Comparison (Scalability Analysis)")
    graph_3_efficiency_comparison()
    
    print("4. Comparative Speedup (All Implementations)")
    graph_4_comparative_speedup()
    
    print("\n" + "="*70)
    print("✓ Successfully generated 4 critical graphs!")
    print("="*70 + "\n")
    
    print("These graphs support:")
    print("  • Section 3.2: Speedup Analysis (Graphs 1, 4)")
    print("  • Section 3.3: Efficiency Analysis (Graph 3)")
    print("  • Section 3.4: Performance Bottlenecks (Graph 2)")
    print("  • Section 3.6: Overhead Analysis (Graph 1)")
    print("\nNext: Update your actual data at the top of this script, then run again!")
    print()


if __name__ == "__main__":
    main()
