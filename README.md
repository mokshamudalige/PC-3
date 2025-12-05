# Gaussian Blur Parallel Implementations

**Course:** SE3082 - Parallel Computing  
**Student:** P.D.M.D.M. Mudalige  
**Student ID:** IT23372344  
**Date:** December 5, 2025

---

## Project Overview

This project implements Gaussian blur image filtering using three parallel programming paradigms:
- **OpenMP** (Shared Memory)
- **MPI** (Distributed Memory)
- **CUDA** (GPU Acceleration)

**Algorithm:** 3×3 Gaussian convolution kernel applied to 1000×1000 grayscale images

---

## Prerequisites

### Software Requirements
- **GCC 13.3.0+** with OpenMP support
- **Open MPI 4.1.6+**
- **NVIDIA CUDA Toolkit 12.0+** (for CUDA implementation)
- **Python 3.x** with matplotlib and numpy (for graph generation)

### Installation Commands (Ubuntu/WSL)
```bash
# Install GCC with OpenMP
sudo apt update
sudo apt install build-essential

# Install OpenMPI
sudo apt install openmpi-bin openmpi-common libopenmpi-dev

# Install Python dependencies
pip3 install matplotlib numpy
```

---

## Compilation Instructions

### Serial Implementation
```bash
gcc -O3 -march=native serial.c -o serial
```

### OpenMP Implementation
```bash
cd openmp_performance
gcc -fopenmp -O3 -march=native gaussian_blur_openmp.c -o gaussian_blur_openmp
```

### MPI Implementation
```bash
cd mpi_performance
mpicc -O3 -march=native gaussian_blur_mpi.c -o gaussian_blur_mpi
```

### CUDA Implementation (Google Colab or NVIDIA GPU)
```bash
cd cuda_performance
nvcc -arch=sm_75 -O3 gaussian_blur_cuda.cu -o gaussian_blur_cuda
```

**Note:** Replace `-arch=sm_75` with your GPU's compute capability:
- Tesla K80: `-arch=sm_37`
- Tesla P100: `-arch=sm_60`
- Tesla V100: `-arch=sm_70`
- Tesla T4: `-arch=sm_75`
- A100: `-arch=sm_80`

---

## Execution Instructions

### Serial Baseline
```bash
./serial
```

### OpenMP (Vary Thread Count)
```bash
# Using command-line argument
./gaussian_blur_openmp 1    # 1 thread
./gaussian_blur_openmp 2    # 2 threads
./gaussian_blur_openmp 4    # 4 threads
./gaussian_blur_openmp 8    # 8 threads

# OR using environment variable
export OMP_NUM_THREADS=8
./gaussian_blur_openmp
```

### MPI (Vary Process Count)
```bash
mpirun -np 1 ./gaussian_blur_mpi     # 1 process
mpirun -np 2 ./gaussian_blur_mpi     # 2 processes
mpirun -np 4 ./gaussian_blur_mpi     # 4 processes
mpirun -np 8 ./gaussian_blur_mpi     # 8 processes (may fail on single node)
```

### CUDA (Vary Block Size and Kernel Type)
```bash
# Syntax: ./gaussian_blur_cuda <block_size> <use_shared_memory>
# block_size: 8, 16, or 32
# use_shared_memory: 0 (basic) or 1 (shared memory)

# Basic kernel with different block sizes
./gaussian_blur_cuda 8 0     # 8×8 blocks, basic kernel
./gaussian_blur_cuda 16 0    # 16×16 blocks, basic kernel
./gaussian_blur_cuda 32 0    # 32×32 blocks, basic kernel

# Shared memory kernel with different block sizes
./gaussian_blur_cuda 8 1     # 8×8 blocks, shared memory
./gaussian_blur_cuda 16 1    # 16×16 blocks, shared memory
./gaussian_blur_cuda 32 1    # 32×32 blocks, shared memory
```

---

## Troubleshooting

### OpenMP: Thread count doesn't match
```bash
# Check available threads
echo $OMP_NUM_THREADS

# Force thread count
export OMP_NUM_THREADS=8
```

### MPI: "not enough slots available"
```bash
# Allow oversubscription
mpirun --oversubscribe -np 8 ./gaussian_blur_mpi

# Or specify slots
mpirun --host localhost:8 -np 8 ./gaussian_blur_mpi
```

### CUDA: Compilation errors
```bash
# Check CUDA installation
nvcc --version

# Check GPU compute capability
nvidia-smi

# Use correct architecture flag
nvcc -arch=sm_XX ...  # Replace XX with your GPU's compute capability
```

---

## Contact

**Student:** P.D.M.D.M. Mudalige  
**ID:** IT23372344  
**Course:** SE3082 - Parallel Computing  
**Institution:** Sri Lanka Institute of Information Technology

