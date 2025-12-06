"""
MPI Performance Analysis - Graph Generation
Generates:
1. Processes vs Execution Time
2. Processes vs Speedup
"""

import matplotlib.pyplot as plt
import numpy as np



processes = [1, 2, 4, 8, 16]

# Example data - REPLACE with your actual execution times (in seconds)
execution_times = [0.055, 0.032, 0.020, 0.015, 0.013]  # REPLACE WITH YOUR DATA

# Serial baseline time (for speedup calculation)
serial_time = execution_times[0]  # Usually the 1-process time

# Calculate speedup
speedup = [serial_time / t for t in execution_times]

# Calculate efficiency
efficiency = [speedup[i] / processes[i] * 100 for i in range(len(processes))]

# ============================================
# GRAPH 1: Processes vs Execution Time
# ============================================
plt.figure(figsize=(10, 6))
plt.plot(processes, execution_times, marker='s', linewidth=2, markersize=8, color='#06A77D')
plt.xlabel('Number of Processes', fontsize=12, fontweight='bold')
plt.ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
plt.title('MPI: Processes vs Execution Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(processes)

# Add value labels on points
for i, (x, y) in enumerate(zip(processes, execution_times)):
    plt.annotate(f'{y:.4f}s', 
                xy=(x, y), 
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=9)

plt.tight_layout()
plt.savefig('mpi_processes_vs_time.png', dpi=300, bbox_inches='tight')
print("✅ Generated: mpi_processes_vs_time.png")
plt.close()

# ============================================
# GRAPH 2: Processes vs Speedup
# ============================================
plt.figure(figsize=(10, 6))

# Plot actual speedup
plt.plot(processes, speedup, marker='s', linewidth=2, markersize=8, 
         color='#D00000', label='Actual Speedup')

# Plot ideal speedup (linear)
ideal_speedup = processes
plt.plot(processes, ideal_speedup, linestyle='--', linewidth=2, 
         color='#F77F00', label='Ideal Speedup (Linear)')

plt.xlabel('Number of Processes', fontsize=12, fontweight='bold')
plt.ylabel('Speedup', fontsize=12, fontweight='bold')
plt.title('MPI: Processes vs Speedup', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(processes)
plt.legend(fontsize=10)

# Add value labels on points
for i, (x, y) in enumerate(zip(processes, speedup)):
    plt.annotate(f'{y:.2f}x', 
                xy=(x, y), 
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=9)

plt.tight_layout()
plt.savefig('mpi_processes_vs_speedup.png', dpi=300, bbox_inches='tight')
print("✅ Generated: mpi_processes_vs_speedup.png")
plt.close()

# ============================================
# BONUS: Performance Summary Table
# ============================================
plt.figure(figsize=(10, 5))
plt.axis('off')

table_data = []
table_data.append(['Processes', 'Time (s)', 'Speedup', 'Efficiency (%)'])
for i in range(len(processes)):
    table_data.append([
        str(processes[i]),
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
    table[(0, i)].set_facecolor('#06A77D')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    color = '#E8F8F5' if i % 2 == 0 else 'white'
    for j in range(4):
        table[(i, j)].set_facecolor(color)

plt.title('MPI Performance Summary', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('mpi_performance_table.png', dpi=300, bbox_inches='tight')
print("✅ Generated: mpi_performance_table.png")
plt.close()

print("\n" + "="*60)
print("MPI Graph Generation Complete!")
print("="*60)
print("Generated files:")
print("  1. mpi_processes_vs_time.png")
print("  2. mpi_processes_vs_speedup.png")
print("  3. mpi_performance_table.png (bonus)")
print("\n⚠️  Remember to replace the example data with your actual test results!")
print("="*60)
