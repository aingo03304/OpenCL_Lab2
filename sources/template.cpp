#include <CL/opencl.h>
#include <gputk.h>

//@@ Compute C = A * B
const char *kernelSource = "";


int main(int argc, char **argv) {
  gpuTKArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);

  //@@ Set numCRows and numCColumns
  numCRows    = 0;
  numCColumns = 0;

  //@@ Allocate the hostC matrix
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  gpuTKLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  //@@ Initialize the workgroup dimensions

  //@@ Bind to platform

  //@@ Get ID for the device

  //@@ Create a context

  //@@ Create a command queue

  //@@ Create the compute program from the source buffer

  //@@ Build the program executable

  //@@ Create the compute kernel in the program we wish to run

  //@@ Create the input and output arrays in device memory for our
  //@@ calculation

  //@@ Write our data set into the input array in device memory

  //@@ Set the arguments to our compute kernel

  //@@ Execute the kernel over the entire range of the data set
  

  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here

  //@@ Wait for the command queue to get serviced before reading back results
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  //@@ Read the results from the device

  gpuTKSolution(args, hostC, numCRows, numCColumns);

  // release OpenCL resources 
  clReleaseMemObject(deviceA);
  clReleaseMemObject(deviceB);
  clReleaseMemObject(deviceC);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);


  // release host memory
  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
