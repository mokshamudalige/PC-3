"""
Comparative Analysis - All Implementations
Generates:
1. Execution Time Comparison (Serial, OpenMP, MPI, CUDA)
2. Speedup Comparison (OpenMP, MPI, CUDA vs Serial)
3. Efficiency Comparison
"""

import matplotlib.pyplot as plt
import numpy as np

# ============================================
# INPUT YOUR TEST RESULTS FROM ALL IMPLEMENTATIONS
# ============================================

# Serial baseline
serial_time = 0.050  # REPLACE WITH YOUR ACTUAL SERIAL TIME

# OpenMP best time (use the fastest from your tests, typically 16 threads)
openmp_best_threads = 16
openmp_best_time = 0.008  # REPLACE WITH YOUR DATA

# MPI best time (use the fastest from your tests, typically 16 processes)
mpi_best_processes = 16
mpi_best_time = 0.013  # REPLACE WITH YOUR DATA

# CUDA best time (use the fastest configuration - typically 16x16 Shared)
cuda_best_config = "16x16 Shared"
cuda_best_time = 0.0008  # REPLACE WITH YOUR DATA

# ============================================
# Data preparation
# ============================================
implementations = ['Serial', 'OpenMP\n(16 threads)', 'MPI\n(16 processes)', 'CUDA\n(16x16 Shared)']
execution_times = [serial_time, openmp_best_time, mpi_best_time, cuda_best_time]
colors = ['#6C757D', '#2E86AB', '#06A77D', '#7209B7']

# Calculate speedups (relative to serial)
speedups = [1.0]  # Serial speedup is always 1x
speedups.extend([serial_time / t for t in [openmp_best_time, mpi_best_time, cuda_best_time]])

# ============================================
# GRAPH 1: Execution Time Comparison
# ============================================
plt.figure(figsize=(12, 7))

bars = plt.bar(implementations, execution_times, color=colors, alpha=0.8, width=0.6)
plt.ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
plt.title('Performance Comparison: Execution Time\n(Serial vs Parallel Implementations)', 
         fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, time in zip(bars, execution_times):
    plt.text(bar.get_x() + bar.get_width()/2., time,
            f'{time:.6f}s',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add performance ranking annotation
fastest_idx = execution_times.index(min(execution_times))
plt.text(fastest_idx, execution_times[fastest_idx] * 1.1,
        '⭐ Fastest',
        ha='center', fontsize=10, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig('comparative_execution_time.png', dpi=300, bbox_inches='tight')
print("✅ Generated: comparative_execution_time.png")
plt.close()

# ============================================
# GRAPH 2: Speedup Comparison
# ============================================
plt.figure(figsize=(12, 7))

# Plot speedup bars (excluding serial which is 1x)
parallel_implementations = implementations[1:]
parallel_speedups = speedups[1:]
parallel_colors = colors[1:]

bars = plt.bar(parallel_implementations, parallel_speedups, 
              color=parallel_colors, alpha=0.8, width=0.6)
plt.ylabel('Speedup (vs Serial)', fontsize=12, fontweight='bold')
plt.title('Performance Comparison: Speedup\n(Relative to Serial Implementation)', 
         fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add baseline line at 1x
plt.axhline(y=1, color='red', linestyle='--', linewidth=2, 
           label='Serial Baseline (1x)', alpha=0.5)
plt.legend(fontsize=10)

# Add value labels on bars
for bar, speedup in zip(bars, parallel_speedups):
    plt.text(bar.get_x() + bar.get_width()/2., speedup,
            f'{speedup:.1f}x',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Highlight best speedup
best_speedup_idx = parallel_speedups.index(max(parallel_speedups))
plt.text(best_speedup_idx, parallel_speedups[best_speedup_idx] * 1.05,
        '⭐ Best Speedup',
        ha='center', fontsize=10, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig('comparative_speedup.png', dpi=300, bbox_inches='tight')
print("✅ Generated: comparative_speedup.png")
plt.close()

# ============================================
# GRAPH 3: Combined Line Chart (Time Trend)
# ============================================
plt.figure(figsize=(12, 7))

x_pos = np.arange(len(implementations))
plt.plot(x_pos, execution_times, marker='o', linewidth=3, 
        markersize=12, color='#E63946', label='Execution Time')

plt.xlabel('Implementation', fontsize=12, fontweight='bold')
plt.ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
plt.title('Performance Trend Across Implementations', fontsize=14, fontweight='bold')
plt.xticks(x_pos, implementations)
plt.grid(True, alpha=0.3)

# Fill area under curve
plt.fill_between(x_pos, execution_times, alpha=0.3, color='#E63946')

# Add value labels
for i, (x, y) in enumerate(zip(x_pos, execution_times)):
    plt.annotate(f'{y:.6f}s', 
                xy=(x, y), 
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=9,
                fontweight='bold')

plt.tight_layout()
plt.savefig('comparative_trend.png', dpi=300, bbox_inches='tight')
print("✅ Generated: comparative_trend.png")
plt.close()

# ============================================
# GRAPH 4: Performance Improvement (%)
# ============================================
plt.figure(figsize=(12, 7))

# Calculate improvement percentage vs serial
improvements = [(serial_time - t) / serial_time * 100 for t in execution_times]
parallel_improvements = improvements[1:]

bars = plt.bar(parallel_implementations, parallel_improvements, 
              color=parallel_colors, alpha=0.8, width=0.6)
plt.ylabel('Performance Improvement (%)', fontsize=12, fontweight='bold')
plt.title('Performance Improvement Over Serial\n(Percentage Reduction in Execution Time)', 
         fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, improvement in zip(bars, parallel_improvements):
    plt.text(bar.get_x() + bar.get_width()/2., improvement,
            f'{improvement:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('comparative_improvement.png', dpi=300, bbox_inches='tight')
print("✅ Generated: comparative_improvement.png")
plt.close()

# ============================================
# BONUS: Comprehensive Summary Table
# ============================================
plt.figure(figsize=(14, 6))
plt.axis('off')

table_data = []
table_data.append(['Implementation', 'Configuration', 'Time (s)', 'Speedup', 'Improvement', 'Rank'])

configs = ['N/A', f'{openmp_best_threads} threads', f'{mpi_best_processes} processes', cuda_best_config]
ranks = ['4', '3', '2', '1']  # Adjust based on your actual results

for i, impl in enumerate(implementations):
    table_data.append([
        impl.replace('\n', ' '),
        configs[i],
        f'{execution_times[i]:.6f}',
        f'{speedups[i]:.2f}x',
        f'{improvements[i]:.1f}%',
        ranks[i] if i > 0 else 'Baseline'
    ])

table = plt.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.18, 0.2, 0.15, 0.15, 0.17, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for i in range(6):
    table[(0, i)].set_facecolor('#2C3E50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color code rows by performance
row_colors = ['#E8E8E8', '#C8E6C9', '#A5D6A7', '#81C784']  # Grey, light green, green, dark green
for i in range(1, len(table_data)):
    for j in range(6):
        table[(i, j)].set_facecolor(row_colors[i-1])

plt.title('Comprehensive Performance Comparison Summary', 
         fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('comparative_summary_table.png', dpi=300, bbox_inches='tight')
print("✅ Generated: comparative_summary_table.png")
plt.close()

# ============================================
# Print Summary Statistics
# ============================================
print("\n" + "="*70)
print("COMPARATIVE ANALYSIS SUMMARY")
print("="*70)
print(f"\n{'Implementation':<25} {'Time (s)':<12} {'Speedup':<12} {'Improvement'}")
print("-"*70)
for i, impl in enumerate(implementations):
    print(f"{impl.replace(chr(10), ' '):<25} {execution_times[i]:<12.6f} {speedups[i]:<12.2f} {improvements[i]:.1f}%")

print("\n" + "="*70)
print("KEY FINDINGS:")
print("="*70)
fastest = implementations[execution_times.index(min(execution_times))].replace('\n', ' ')
best_speedup_impl = implementations[1:][speedups[1:].index(max(speedups[1:]))].replace('\n', ' ')
print(f"🏆 Fastest Implementation: {fastest}")
print(f"⚡ Best Speedup: {best_speedup_impl} ({max(speedups[1:]):.2f}x)")
print(f"📊 CUDA vs OpenMP: {(openmp_best_time/cuda_best_time):.1f}x faster")
print(f"📊 CUDA vs MPI: {(mpi_best_time/cuda_best_time):.1f}x faster")
print("="*70)

print("\n✅ Comparative Analysis Complete!")
print("\nGenerated files:")
print("  1. comparative_execution_time.png")
print("  2. comparative_speedup.png")
print("  3. comparative_trend.png")
print("  4. comparative_improvement.png")
print("  5. comparative_summary_table.png")
print("\n⚠️  Remember to replace the example data with your actual test results!")
print("="*70)
