"""
Master Script - Generate All Performance Graphs
Runs all graph generation scripts in sequence
"""

import subprocess
import sys

print("="*70)
print("PERFORMANCE EVALUATION - GRAPH GENERATION SUITE")
print("="*70)
print("\nThis will generate all required graphs for Part B (25 marks)")
print("\n  IMPORTANT: Before running, make sure you have:")
print("   1. Completed all tests (Serial, OpenMP, MPI, CUDA)")
print("   2. Updated the data in each script with your actual results")
print("   3. Installed matplotlib: pip install matplotlib numpy")
print("\n" + "="*70)

response = input("\nHave you updated all scripts with your test data? (yes/no): ")
if response.lower() != 'yes':
    print("\n Please update the data in each script first!")
    print("   Edit these files:")
    print("   - generate_openmp_graphs.py")
    print("   - generate_mpi_graphs.py")
    print("   - generate_cuda_graphs.py")
    print("   - generate_comparative_graphs.py")
    sys.exit(0)

print("\n" + "="*70)
print("Starting graph generation...")
print("="*70 + "\n")

scripts = [
    ("../openmp_performance/generate_openmp_graphs.py", "OpenMP Graphs (7-8)"),
    ("../mpi_performance/generate_mpi_graphs.py", "MPI Graphs (8)"),
    ("../cuda_performance/generate_cuda_graphs.py", "CUDA Graphs (9)"),
    ("../comparative_analysis/generate_comparative_graphs.py", "Comparative Analysis (10)")
]

total_graphs = 0
failed_scripts = []

for script, description in scripts:
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Script: {script}")
    print("="*70)
    
    try:
        result = subprocess.run([sys.executable, script], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        
        # Count generated graphs from output
        graph_count = result.stdout.count(" Generated:")
        total_graphs += graph_count
        print(f" {description} completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f" Error running {script}:")
        print(e.stderr)
        failed_scripts.append(script)
    except FileNotFoundError:
        print(f" Script not found: {script}")
        failed_scripts.append(script)

print("\n" + "="*70)
print("GENERATION COMPLETE!")
print("="*70)
print(f"\nTotal graphs generated: {total_graphs}")

if failed_scripts:
    print(f"\n Failed scripts ({len(failed_scripts)}):")
    for script in failed_scripts:
        print(f"   - {script}")
else:
    print("\n All scripts executed successfully!")

print("\n" + "="*70)
print("GENERATED FILES:")
print("="*70)
print("\nOpenMP Graphs (Requirement 7):")
print("  ✓ openmp_threads_vs_time.png")
print("  ✓ openmp_threads_vs_speedup.png")
print("  ✓ openmp_performance_table.png (bonus)")

print("\nMPI Graphs (Requirement 8):")
print("  ✓ mpi_processes_vs_time.png")
print("  ✓ mpi_processes_vs_speedup.png")
print("  ✓ mpi_performance_table.png")

print("\nCUDA Graphs (Requirement 9):")
print("  ✓ cuda_blocksize_vs_time.png")
print("  ✓ cuda_blocksize_vs_speedup.png")
print("  ✓ cuda_kernel_comparison.png")
print("  ✓ cuda_performance_table.png")

print("\nComparative Analysis (Requirement 10):")
print("  ✓ comparative_execution_time.png")
print("  ✓ comparative_speedup.png")
print("  ✓ comparative_trend.png")
print("  ✓ comparative_improvement.png")
print("  ✓ comparative_summary_table.png ")

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("1. Review all generated PNG files")
print("2. Insert graphs into your report")
print("3. Write analysis for each graph")
print("4. Discuss findings and conclusions")
print("="*70)
