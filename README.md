# SIMTPeridigm
[Peridigm](https://github.com/peridigm/peridigm) is an open-source computational peridynamics code developed at Sandia National Laboratories for massively-parallel multi-physics simulations. It has been applied primarily to problems in solid mechanics involving pervasive material failure. Peridigm is a C++ code utilizing foundational software components from Sandia's Trilinos project and is fully compatible with the Cubit mesh generator and Paraview visualization code. </br>
In order to provide high-performance long-term and large-scale Peridynamics simulations, we redesign the Peridigm to run Peridigm on SIMT accelerators e.g. GPUs.
## Getting started
Peridigm is a C++ code intended for use on Mac and Linux operating systems. Both Peridigm and the Trilinos libraries it depends on should be built using MPI compilers and the CMake build system. The Peridigm test harness requires python. The build process has been tested using gcc and Intel compilers, Open MPI, and MPICH. To make it run on the SIMT accelerators, CUDA or HIP are also needed.  The steps below should be followed in order, beginning with the installation of the required third-party libraries, installing the original version of Peridigm, and installing our code.  </br>
To install our Peridigm, users need to install third-party packages and the original version of Peridigm. Later compiled our code to generate a dynamic-linked library. The example script for installing is in `build/` and the example scripts for compiled code are in `src/gpu_kenel/nvcc_src` and `src/gpu_kernel/hip_src`.
* [Installing Third-Party Packages and Libraries](https://github.com/peridigm/peridigm/blob/master/doc/InstallingThirdPartyLibs.md)
* [Building Peridigm](https://github.com/peridigm/peridigm/blob/master/doc/BuildingPeridigm.md)
* Installing our code</br>
  * cd `src/gpu_kenel/nvcc_src`(for cuda users) or cd  `src/gpukernel/hip_src` (for HIP users)
  * editting an installation script refer to `lddlib.sh.example`
  * sh lddlib.sh 
 * Running Simulations with Peridigm
   * Serial： Peridigm fragmenting_cylinder.yaml
   * Parallel:  mpirun -np n Peridigm fragmenting_cylinder.yaml
