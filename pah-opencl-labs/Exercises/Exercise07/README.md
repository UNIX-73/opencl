Exercise 7 - Matrix Multiplication using local memory
===============================

Goal
----
* Use local memory to minimize memory movement costs and optimize performance of your matrix multiplication program.

Procedure
---------
* Start with the provided matrix multiplication OpenCL code.
* Complete the kernel *C\_row\_priv\_bloc.cl*: row per work-item, A private, B local.
* In addition, complete the blank parts in the host code *matmul* to run the kernel.

Expected output
---------------
* A message verifying the code has completed successfully.
* Report the runtime and the MFLOPS.
