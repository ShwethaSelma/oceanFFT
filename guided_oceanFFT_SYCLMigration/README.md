# `OceanFFT` Sample

`OceanFFT` is a simulation of ocean waves/height field using Fast Fourier Transform (FFT). FFT is a way to transform a set of data from one frame of reference to another. This sample is implemented using SYCL* by migrating code from original CUDA source code for offloading computations to a GPU/CPU.

> **Note**: This sample is migrated from NVIDIA CUDA sample. See the [OceanFFT](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries/oceanFFT) sample in the NVIDIA/cuda-samples GitHub.

| Property                       | Description
|:---                               |:---
| What you will learn               | How to begin migrating CUDA to SYCL*
| Time to complete                  | 15 minutes


## Purpose

Ocean is made of many waves all added together. The main principle of Ocean rendering is that it can be modelled as sum of infinite waves at different amplitudes travelling in different directions. So in order to simulate ocean waves, we need to be able to sample the height at any 2D point in the world. First step is to generate an image of the Ocean in the frequency domain and then by running it through the FFT, get the data in the spatial domain. Once the data is in spatial domain get the height displacement value.

This OceanFFT SYCL application demonstrates the ocean wave simulation using oneMKL DFT  library and uses image processing key concept. Intel® oneAPI Math Kernel Library (oneMKL) provides a DPC++ interface for computing a discrete Fourier transform through the fast Fourier transform algorithm. The DPC++ interface emulates the usage model of the oneMKL C and Fortran Discrete Fourier Transform Interface (DFTI).


This sample contains four versions:

| Folder Name                          | Description
|:---                                  |:---
| 01_sycl_dpct_output                  | Contains output of Intel® DPC++ Compatibility Tool used to migrate SYCL-compliant code from CUDA code, this SYCL code has some unmigrated code which has to be manually fixed to get full functionality, the code does not functionally work.
| 02_sycl_dpct_migrated                | Contains Intel® DPC++ Compatibility Tool migrated SYCL code from CUDA code with manual changes done to fix the unmigrated code to work functionally.
| 03_sycl_migrated                     | Contains manually migrated SYCL code from CUDA code.

## Prerequisites
| Property                       | Description
|:---                               |:---
| OS                                | Ubuntu* 20.04
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler

## Key Implementation Details

In the OceanFFT sample ocean waves/height field is simulated using FFT. This sample uses oneMKL DFT calls to synthesize and render an ocean surface in real-time. FFT transfers the data from the frequency domain to spatial domain to obtain the height displacement value. 

The implementation involves creation of FFT data using the `mkl::dft::descriptor` api and initiates it with default configuration. Next step is to generate wave spectrum frequency domain based on initial heightfield and dispersion relationship, and then the data is ran through inverse FFT(`dft::compute_forward / dft::compute_backward`) to get the data in spatial domain from frequency domain. Based on the FFT output the height field values are updated. Final step is to calculate the slope for shading by partial difference in spacial domain.


## Build the `OceanFFT` Sample for CPU and GPU

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `$ bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
>For more information on environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### On Linux*
Perform the following steps:
1. Change to the `guided_oceanFFT_SYCLMigration` directory.
2. Build the program.
   ```
   $ mkdir build
   $ cd build
   $ cmake ..
   $ make
   ```

   By default, these commands build the `02_sycl_dpct_migrated` and `03_sycl_migrated` versions of the program.

If an error occurs, you can get more details by running `make` with the `VERBOSE=1` argument:
```
make VERBOSE=1
```

#### Troubleshooting
If you receive an error message, troubleshoot the problem using the Diagnostics Utility for Intel&reg; oneAPI Toolkits. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors and other issues. See [Diagnostics Utility for Intel&reg; oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## Run the `OceanFFT` Sample
In all cases, you can run the programs for CPU and GPU. The run commands indicate the device target.
1. Run `02_sycl_dpct_migrated` for CPU and GPU.
    ```
    make run_sdm_cpu
    make run_sdm_gpu
    ```
    
2. Run `03_sycl_migrated` for CPU and GPU.
    ```
    make run_cpu
    make run_gpu
    ```
    

### Run the `OceanFFT` Sample in Intel&reg; DevCloud

When running a sample in the Intel&reg; DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel&reg; oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).

#### Build and Run Samples in Batch Mode (Optional)

You can submit build and run jobs through a Portable Bash Script (PBS). A job is a script that submitted to PBS through the `qsub` utility. By default, the `qsub` utility does not inherit the current environment variables or your current working directory, so you might need to submit jobs to configure the environment variables. To indicate the correct working directory, you can use either absolute paths or pass the `-d \<dir\>` option to `qsub`.

1. Open a terminal on a Linux* system.
2. Log in to Intel&reg; DevCloud.
    ```
    ssh devcloud
    ```
3. Download the samples.
    ```
    git clone https://github.com/oneapi-src/oneAPI-samples.git
    ```
4. Change to the `guided_oceanFFT_SYCLMigration` directory.
    ```
    cd ~/oneAPI-samples/DirectProgramming/DPC++/SpectralMethods/guided_oceanFFT_SYCLMigration
    ```
5. Configure the sample for a GPU node using `qsub`.
    ```
    qsub  -I  -l nodes=1:gpu:ppn=2 -d .
    ```
    - `-I` (upper case I) requests an interactive session.
    - `-l nodes=1:gpu:ppn=2` (lower case L) assigns one full GPU node.
    - `-d .` makes the current folder as the working directory for the task.
6. Perform build steps as you would on Linux.
7. Run the sample.
8. Clean up the project files.
    ```
    make clean
    ```
9. Disconnect from the Intel&reg; DevCloud.
    ```
    exit
    ```

### Example Output
The following example is for `03_sycl_migrated` for CPU(MKL DFT is optimized for CPU devices) on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz.
```
Compute Capability: 3.0sdkDumpBin: <spatialDomain.bin>
> compareBin2Bin <float> nelements=65536, epsilon=0.10, threshold=0.15
   src_file <spatialDomain.bin>, size=262144 bytes
   ref_file <./data/ref_spatialDomain.bin>, size=262144 bytes
  OK
sdkDumpBin: <slopeShading.bin>
> compareBin2Bin <float> nelements=131072, epsilon=0.10, threshold=0.15
   src_file <slopeShading.bin>, size=524288 bytes
   ref_file <./data/ref_slopeShading.bin>, size=524288 bytes
  OK
Processing time : 213.310989 (ms)
```

The following example is for `03_sycl_migrated` for GPU on Intel(R) UHD Graphics P630 [0x3e96].
```
Compute Capability: 1.3sdkDumpBin: <spatialDomain.bin>
> compareBin2Bin <float> nelements=65536, epsilon=0.10, threshold=0.15
   src_file <spatialDomain.bin>, size=262144 bytes
   ref_file <./data/ref_spatialDomain.bin>, size=262144 bytes
  OK
sdkDumpBin: <slopeShading.bin>
> compareBin2Bin <float> nelements=131072, epsilon=0.10, threshold=0.15
   src_file <slopeShading.bin>, size=524288 bytes
   ref_file <./data/ref_slopeShading.bin>, size=524288 bytes
  OK
Processing time : 2295.610840 (ms)
```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
