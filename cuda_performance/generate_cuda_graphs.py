"""
CUDA Performance Analysis - Graph Generation
Generates:
1. Block Size vs Execution Time (for Basic and Shared kernels)
2. Block Size vs Speedup (comparing Basic and Shared to Serial)
3. Kernel Comparison (Basic vs Shared Memory)
"""

import matplotlib.pyplot as plt
import numpy as np


# Block sizes tested
block_sizes = ['8x8', '16x16', '32x32']
block_sizes_numeric = [8, 16, 32]  # For plotting

# Example data - REPLACE with your actual execution times (in seconds)
# Times for Basic Kernel
basic_kernel_times = [0.0015, 0.0012, 0.0014]  
# Times for Shared Memory Kernel
shared_kernel_times = [0.0010, 0.0008, 0.0011]  

# Serial baseline time (from your serial.c or OpenMP 1-thread test)
serial_time = 0.050  
# Calculate speedup
basic_speedup = [serial_time / t for t in basic_kernel_times]
shared_speedup = [serial_time / t for t in shared_kernel_times]

# ============================================
# GRAPH 1: Block Size vs Execution Time
# ============================================
plt.figure(figsize=(10, 6))

x_pos = np.arange(len(block_sizes))
width = 0.35

bars1 = plt.bar(x_pos - width/2, basic_kernel_times, width, 
               label='Basic Kernel', color='#7209B7', alpha=0.8)
bars2 = plt.bar(x_pos + width/2, shared_kernel_times, width,
               label='Shared Memory Kernel', color='#3A0CA3', alpha=0.8)

plt.xlabel('Block Size', fontsize=12, fontweight='bold')
plt.ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
plt.title('CUDA: Block Size vs Execution Time', fontsize=14, fontweight='bold')
plt.xticks(x_pos, block_sizes)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.6f}s',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('cuda_blocksize_vs_time.png', dpi=300, bbox_inches='tight')
print("✅ Generated: cuda_blocksize_vs_time.png")
plt.close()

# ============================================
# GRAPH 2: Block Size vs Speedup
# ============================================
plt.figure(figsize=(10, 6))

plt.plot(block_sizes_numeric, basic_speedup, marker='o', linewidth=2, 
         markersize=8, color='#7209B7', label='Basic Kernel')
plt.plot(block_sizes_numeric, shared_speedup, marker='s', linewidth=2, 
         markersize=8, color='#3A0CA3', label='Shared Memory Kernel')

plt.xlabel('Block Size', fontsize=12, fontweight='bold')
plt.ylabel('Speedup (vs Serial)', fontsize=12, fontweight='bold')
plt.title('CUDA: Block Size vs Speedup', fontsize=14, fontweight='bold')
plt.xticks(block_sizes_numeric, block_sizes)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Add value labels on points
for i, (x, y) in enumerate(zip(block_sizes_numeric, basic_speedup)):
    plt.annotate(f'{y:.1f}x', 
                xy=(x, y), 
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=9)

for i, (x, y) in enumerate(zip(block_sizes_numeric, shared_speedup)):
    plt.annotate(f'{y:.1f}x', 
                xy=(x, y), 
                textcoords="offset points",
                xytext=(0, -15),
                ha='center',
                fontsize=9)

plt.tight_layout()
plt.savefig('cuda_blocksize_vs_speedup.png', dpi=300, bbox_inches='tight')
print("✅ Generated: cuda_blocksize_vs_speedup.png")
plt.close()

# ============================================
# GRAPH 3: Kernel Comparison (Basic vs Shared)
# ============================================
plt.figure(figsize=(10, 6))

# Calculate average performance for each kernel
avg_basic_time = np.mean(basic_kernel_times)
avg_shared_time = np.mean(shared_kernel_times)
improvement = ((avg_basic_time - avg_shared_time) / avg_basic_time) * 100

kernels = ['Basic Kernel', 'Shared Memory Kernel']
avg_times = [avg_basic_time, avg_shared_time]
colors = ['#7209B7', '#3A0CA3']

bars = plt.bar(kernels, avg_times, color=colors, alpha=0.8, width=0.6)
plt.ylabel('Average Execution Time (seconds)', fontsize=12, fontweight='bold')
plt.title('CUDA: Kernel Performance Comparison', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, time in zip(bars, avg_times):
    plt.text(bar.get_x() + bar.get_width()/2., time,
            f'{time:.6f}s',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add improvement annotation
plt.text(0.5, max(avg_times) * 0.9, 
        f'Shared Memory\n{improvement:.1f}% faster',
        ha='center', fontsize=11, 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('cuda_kernel_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Generated: cuda_kernel_comparison.png")
plt.close()

# ============================================
# BONUS: Performance Summary Table
# ============================================
plt.figure(figsize=(12, 6))
plt.axis('off')

table_data = []
table_data.append(['Block Size', 'Basic Time (s)', 'Shared Time (s)', 
                  'Basic Speedup', 'Shared Speedup', 'Improvement'])
for i in range(len(block_sizes)):
    improvement_pct = ((basic_kernel_times[i] - shared_kernel_times[i]) / basic_kernel_times[i]) * 100
    table_data.append([
        block_sizes[i],
        f'{basic_kernel_times[i]:.6f}',
        f'{shared_kernel_times[i]:.6f}',
        f'{basic_speedup[i]:.1f}x',
        f'{shared_speedup[i]:.1f}x',
        f'{improvement_pct:.1f}%'
    ])

table = plt.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.15, 0.18, 0.18, 0.16, 0.16, 0.17])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(6):
    table[(0, i)].set_facecolor('#7209B7')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    color = '#F3E5F5' if i % 2 == 0 else 'white'
    for j in range(6):
        table[(i, j)].set_facecolor(color)

plt.title('CUDA Performance Summary', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('cuda_performance_table.png', dpi=300, bbox_inches='tight')
print("✅ Generated: cuda_performance_table.png")
plt.close()

print("\n" + "="*60)
print("CUDA Graph Generation Complete!")
print("="*60)
print("Generated files:")
print("  1. cuda_blocksize_vs_time.png")
print("  2. cuda_blocksize_vs_speedup.png")
print("  3. cuda_kernel_comparison.png")
print("  4. cuda_performance_table.png (bonus)")
print("\n⚠️  Remember to replace the example data with your actual test results!")
print("="*60)
