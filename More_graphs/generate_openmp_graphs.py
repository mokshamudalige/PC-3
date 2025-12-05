"""
OpenMP Performance Analysis - Graph Generation
Generates:
1. Threads vs Execution Time
2. Threads vs Speedup
"""

import matplotlib.pyplot as plt
import numpy as np

# ============================================
# INPUT YOUR OPENMP TEST RESULTS HERE
# ============================================
# Replace these with your actual measured values from running the OpenMP tests

threads = [1, 2, 4, 8, 16]

# Example data - REPLACE with your actual execution times (in seconds)
execution_times = [0.050, 0.027, 0.015, 0.010, 0.008]  # REPLACE WITH YOUR DATA

# Serial baseline time (for speedup calculation)
serial_time = execution_times[0]  # Usually the 1-thread time, or run separate serial version

# Calculate speedup
speedup = [serial_time / t for t in execution_times]

# Calculate efficiency
efficiency = [speedup[i] / threads[i] * 100 for i in range(len(threads))]

# ============================================
# GRAPH 1: Threads vs Execution Time
# ============================================
plt.figure(figsize=(10, 6))
plt.plot(threads, execution_times, marker='o', linewidth=2, markersize=8, color='#2E86AB')
plt.xlabel('Number of Threads', fontsize=12, fontweight='bold')
plt.ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
plt.title('OpenMP: Threads vs Execution Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(threads)

# Add value labels on points
for i, (x, y) in enumerate(zip(threads, execution_times)):
    plt.annotate(f'{y:.4f}s', 
                xy=(x, y), 
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=9)

plt.tight_layout()
plt.savefig('openmp_threads_vs_time.png', dpi=300, bbox_inches='tight')
print("✅ Generated: openmp_threads_vs_time.png")
plt.close()

# ============================================
# GRAPH 2: Threads vs Speedup
# ============================================
plt.figure(figsize=(10, 6))

# Plot actual speedup
plt.plot(threads, speedup, marker='o', linewidth=2, markersize=8, 
         color='#A23B72', label='Actual Speedup')

# Plot ideal speedup (linear)
ideal_speedup = threads
plt.plot(threads, ideal_speedup, linestyle='--', linewidth=2, 
         color='#F18F01', label='Ideal Speedup (Linear)')

plt.xlabel('Number of Threads', fontsize=12, fontweight='bold')
plt.ylabel('Speedup', fontsize=12, fontweight='bold')
plt.title('OpenMP: Threads vs Speedup', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(threads)
plt.legend(fontsize=10)

# Add value labels on points
for i, (x, y) in enumerate(zip(threads, speedup)):
    plt.annotate(f'{y:.2f}x', 
                xy=(x, y), 
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=9)

plt.tight_layout()
plt.savefig('openmp_threads_vs_speedup.png', dpi=300, bbox_inches='tight')
print("✅ Generated: openmp_threads_vs_speedup.png")
plt.close()

# ============================================
# BONUS: Performance Summary Table
# ============================================
plt.figure(figsize=(10, 5))
plt.axis('off')

table_data = []
table_data.append(['Threads', 'Time (s)', 'Speedup', 'Efficiency (%)'])
for i in range(len(threads)):
    table_data.append([
        str(threads[i]),
        f'{execution_times[i]:.6f}',
        f'{speedup[i]:.2f}x',
        f'{efficiency[i]:.1f}%'
    ])

table = plt.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.2, 0.25, 0.25, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#2E86AB')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    color = '#E8F4F8' if i % 2 == 0 else 'white'
    for j in range(4):
        table[(i, j)].set_facecolor(color)

plt.title('OpenMP Performance Summary', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('openmp_performance_table.png', dpi=300, bbox_inches='tight')
print("✅ Generated: openmp_performance_table.png")
plt.close()

print("\n" + "="*60)
print("OpenMP Graph Generation Complete!")
print("="*60)
print("Generated files:")
print("  1. openmp_threads_vs_time.png")
print("  2. openmp_threads_vs_speedup.png")
print("  3. openmp_performance_table.png (bonus)")
print("\n⚠️  Remember to replace the example data with your actual test results!")
print("="*60)
