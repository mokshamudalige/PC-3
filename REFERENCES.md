# References Document for Gaussian Blur Parallel Implementations
# SE3082 - Parallel Computing Assignment 03
# Author: P.D.M.D.M.Mudalige (IT23372344)

## COMPLETE REFERENCE LIST

### 1. OPENMP REFERENCES

#### Books:
1. Chapman, B., Jost, G., & Van Der Pas, R. (2008). 
   *Using OpenMP: Portable Shared Memory Parallel Programming* (Vol. 10). 
   MIT Press.
   - Comprehensive guide to OpenMP programming
   - Covers parallel for directives and scheduling strategies
   - ISBN: 978-0262533027

2. Mattson, T., Sanders, B., & Massingill, B. (2004). 
   *Patterns for Parallel Programming*. 
   Addison-Wesley Professional.
   - Design patterns for parallel algorithms
   - ISBN: 978-0321228116

#### Online Resources:
3. OpenMP Architecture Review Board. (2021). 
   *OpenMP Application Programming Interface Version 5.2*.
   Available at: https://www.openmp.org/specifications/
   - Official OpenMP specification
   - Complete API reference

4. Lawrence Livermore National Laboratory. 
   *OpenMP Tutorial*.
   Available at: https://computing.llnl.gov/tutorials/openMP/
   - Practical OpenMP programming guide
   - Examples and best practices

5. Barney, B. 
   *Introduction to Parallel Computing Tutorial*.
   Lawrence Livermore National Laboratory.
   Available at: https://hpc.llnl.gov/documentation/tutorials/introduction-parallel-computing-tutorial

#### Academic Papers:
6. Dagum, L., & Menon, R. (1998). 
   OpenMP: An industry standard API for shared-memory programming. 
   *IEEE Computational Science and Engineering*, 5(1), 46-55.
   - Historical context and design of OpenMP

---

### 2. MPI REFERENCES

#### Books:
7. Gropp, W., Lusk, E., & Skjellum, A. (2014). 
   *Using MPI: Portable Parallel Programming with the Message-Passing Interface* (3rd ed.). 
   MIT Press.
   - Definitive guide to MPI programming
   - ISBN: 978-0262527392

8. Pacheco, P. S. (2011). 
   *An Introduction to Parallel Programming*. 
   Morgan Kaufmann.
   - Covers MPI, Pthreads, and OpenMP
   - Excellent for beginners
   - ISBN: 978-0123742605

#### Online Resources:
9. MPI Forum. (2021). 
   *MPI: A Message-Passing Interface Standard Version 4.0*.
   Available at: https://www.mpi-forum.org/docs/
   - Official MPI specification
   - Complete standard documentation

10. Wes Kendall. 
    *MPI Tutorial*.
    Available at: https://mpitutorial.com/
    - Step-by-step MPI tutorials
    - Code examples for common patterns

11. MPICH Documentation.
    Available at: https://www.mpich.org/documentation/guides/
    - Implementation-specific documentation
    - Performance tuning guides

#### Academic Papers:
12. Walker, D. W., & Dongarra, J. J. (1996). 
    MPI: A standard message passing interface. 
    *Supercomputer*, 12, 56-68.
    - Overview of MPI standard

13. Thakur, R., Rabenseifner, R., & Gropp, W. (2005). 
    Optimization of collective communication operations in MPICH. 
    *The International Journal of High Performance Computing Applications*, 19(1), 49-66.
    - Communication optimization techniques

---

### 3. CUDA REFERENCES

#### Books:
14. Sanders, J., & Kandrot, E. (2010). 
    *CUDA by Example: An Introduction to General-Purpose GPU Programming*. 
    Addison-Wesley Professional.
    - Practical CUDA programming guide
    - ISBN: 978-0131387683

15. Kirk, D. B., & Hwu, W. W. (2016). 
    *Programming Massively Parallel Processors: A Hands-on Approach* (3rd ed.). 
    Morgan Kaufmann.
    - In-depth GPU programming concepts
    - ISBN: 978-0128119860

16. Cheng, J., Grossman, M., & McKercher, T. (2014). 
    *Professional CUDA C Programming*. 
    Wrox.
    - Advanced CUDA programming techniques
    - ISBN: 978-1118739327

#### Online Resources:
17. NVIDIA Corporation. (2023). 
    *CUDA C++ Programming Guide*.
    Available at: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
    - Official CUDA programming guide
    - Complete API reference

18. NVIDIA Corporation. (2023). 
    *CUDA C++ Best Practices Guide*.
    Available at: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
    - Performance optimization strategies
    - Best practices for CUDA development

19. NVIDIA Corporation. 
    *CUDA Toolkit Documentation*.
    Available at: https://docs.nvidia.com/cuda/
    - Comprehensive toolkit documentation
    - Libraries and tools reference

#### Academic Papers:
20. Nickolls, J., Buck, I., Garland, M., & Skadron, K. (2008). 
    Scalable parallel programming with CUDA. 
    *Queue*, 6(2), 40-53.
    - Introduction to CUDA architecture

21. Garland, M., Le Grand, S., Nickolls, J., Anderson, J., Hardwick, J., Morton, S., ... & Volkov, V. (2008). 
    Parallel computing experiences with CUDA. 
    *IEEE Micro*, 28(4), 13-27.
    - Real-world CUDA applications

---

### 4. IMAGE PROCESSING REFERENCES

#### Books:
22. Gonzalez, R. C., & Woods, R. E. (2018). 
    *Digital Image Processing* (4th ed.). 
    Pearson.
    - Comprehensive image processing textbook
    - Covers convolution and filtering
    - ISBN: 978-0133356724

23. Pratt, W. K. (2007). 
    *Digital Image Processing: PIXE* (4th ed.). 
    Wiley-Interscience.
    - Advanced image processing techniques
    - ISBN: 978-0471767770

#### Online Resources:
24. Wikipedia Contributors. 
    *Gaussian blur*.
    Available at: https://en.wikipedia.org/wiki/Gaussian_filter
    - Overview of Gaussian filtering
    - Mathematical foundations

25. Songho. 
    *Image Convolution*.
    Available at: https://www.songho.ca/dsp/convolution/convolution2d.html
    - Interactive convolution tutorial
    - Visual examples

#### Academic Papers:
26. Deriche, R. (1990). 
    Fast algorithms for low-level vision. 
    *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 12(1), 78-87.
    - Efficient convolution algorithms

---

### 5. PARALLEL COMPUTING - GENERAL

#### Books:
27. Wilkinson, B., & Allen, M. (2004). 
    *Parallel Programming: Techniques and Applications Using Networked Workstations and Parallel Computers* (2nd ed.). 
    Pearson.
    - General parallel programming concepts
    - ISBN: 978-0131405639

28. Grama, A., Gupta, A., Karypis, G., & Kumar, V. (2003). 
    *Introduction to Parallel Computing* (2nd ed.). 
    Addison-Wesley.
    - Theoretical foundations of parallel computing
    - ISBN: 978-0201648652

29. Rauber, T., & Rünger, G. (2013). 
    *Parallel Programming: for Multicore and Cluster Systems* (2nd ed.). 
    Springer.
    - Modern parallel programming paradigms
    - ISBN: 978-3642378003

#### Academic Papers:
30. Amdahl, G. M. (1967). 
    Validity of the single processor approach to achieving large scale computing capabilities. 
    *Proceedings of the AFIPS Spring Joint Computer Conference*, 30, 483-485.
    - Amdahl's Law for parallel speedup

31. Gustafson, J. L. (1988). 
    Reevaluating Amdahl's law. 
    *Communications of the ACM*, 31(5), 532-533.
    - Gustafson's Law for scaled speedup

---

### 6. PERFORMANCE EVALUATION

#### Books:
32. Lilja, D. J. (2005). 
    *Measuring Computer Performance: A Practitioner's Guide*. 
    Cambridge University Press.
    - Performance measurement methodologies
    - ISBN: 978-0521646970

#### Academic Papers:
33. Bailey, D. H., Barszcz, E., Barton, J. T., Browning, D. S., Carter, R. L., Dagum, L., ... & Weeratunga, S. (1991). 
    The NAS parallel benchmarks. 
    *The International Journal of Supercomputing Applications*, 5(3), 63-73.
    - Standard parallel benchmarking suite

34. Hoefler, T., & Belli, R. (2015). 
    Scientific benchmarking of parallel computing systems: Twelve ways to tell the masses when reporting performance results. 
    *Proceedings of SC15: International Conference for High Performance Computing, Networking, Storage and Analysis*, 1-12.
    - Best practices for performance reporting

---

### 7. OPTIMIZATION TECHNIQUES

#### Online Resources:
35. Intel Corporation. 
    *Optimization Reference Manual*.
    Available at: https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html
    - CPU optimization techniques

36. NVIDIA Corporation. 
    *GPU Performance Analysis Guide*.
    Available at: https://docs.nvidia.com/nsight-compute/ProfilingGuide/
    - GPU profiling and optimization

#### Academic Papers:
37. Williams, S., Waterman, A., & Patterson, D. (2009). 
    Roofline: An insightful visual performance model for multicore architectures. 
    *Communications of the ACM*, 52(4), 65-76.
    - Performance modeling framework

38. Volkov, V. (2010). 
    Better performance at lower occupancy. 
    *Proceedings of the GPU Technology Conference (GTC)*, 10, 16.
    - CUDA optimization strategies

---

### 8. DOMAIN DECOMPOSITION

#### Academic Papers:
39. Smith, B., Bjørstad, P., & Gropp, W. (2004). 
    *Domain Decomposition: Parallel Multilevel Methods for Elliptic Partial Differential Equations*. 
    Cambridge University Press.
    - Domain decomposition theory
    - ISBN: 978-0521602860

40. Chan, T. F., & Mathew, T. P. (1994). 
    Domain decomposition algorithms. 
    *Acta Numerica*, 3, 61-143.
    - Survey of domain decomposition methods

---

### 9. ADDITIONAL ONLINE RESOURCES

41. Stack Overflow - Parallel Computing Tag
    Available at: https://stackoverflow.com/questions/tagged/parallel-processing
    - Community Q&A for parallel programming

42. GitHub - CUDA Samples
    Available at: https://github.com/NVIDIA/cuda-samples
    - Official CUDA code examples

43. GitHub - OpenMP Examples
    Available at: https://github.com/OpenMP/Examples
    - Official OpenMP example repository

44. Parallel Programming Laboratory, UIUC
    Available at: https://charm.cs.illinois.edu/research/parallel
    - Research and educational materials

---

### 10. HARDWARE SPECIFICATIONS

45. Intel Corporation. 
    *Intel 64 and IA-32 Architectures Optimization Reference Manual*.
    - CPU architecture and optimization

46. NVIDIA Corporation. 
    *NVIDIA GPU Architecture Documentation*.
    Available at: https://developer.nvidia.com/gpu-architecture
    - GPU architecture specifications

---

### CITATION STYLE

This document uses a modified IEEE citation style appropriate for academic 
computing assignments. All references are current as of 2025 and have been 
verified for accessibility.

### USAGE IN ASSIGNMENT

When referencing these materials in your assignment report:
- Cite specific references that directly support your implementation choices
- Include page numbers for books when referencing specific concepts
- For online resources, include access dates if required by your institution
- Reference academic papers when discussing performance metrics or algorithms

### ADDITIONAL NOTES

- All URLs were verified as accessible as of December 2025
- Books listed are standard references in parallel computing education
- Online documentation links point to official sources
- Academic papers are from peer-reviewed journals and conferences
