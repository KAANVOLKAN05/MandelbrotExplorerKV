#ifndef MULTIGPU_H
#define MULTIGPU_H
#include <cuda_runtime.h>
#include <cstddef>


typedef struct
{
    //The Host must know the device ID to decide which GPU to choose
    int deviceID;

    //Host must know how much to offset prior to launching a new GPU
    int xOffset;
    int yOffset;

    //The size of the total grid
    int global_NX;     //NX
    int global_NY;     //NY

    //The size of the grid that we want a single GPU to care about
    //If we were using just a single GPU then globals and locals would be the same
    int local_NX; 
    int local_NY;
    int local_N;         //local_NX * local_NY
    size_t local_Bytes;    // local_N * sizeof(double)
    
    
    //CPU Side arrays
    double *h_lamr;
    double *h_lami;

    //Device Side arrays
    double *d_lamr;
    double *d_lami;

    // Result arrays
    double *h_z;        // Result to copy from an individual GPU
    double *h_mag2;      //Secondary Results to copy from an individual GPU
    double *d_z;        // Result written on the GPU
    double *d_mag2;      // Secondary Results to write on the GPU

    // CUDA stream for this GPU's queued work
    cudaStream_t stream;



} TGPUplan;

#endif