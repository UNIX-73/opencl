Exercise 6 - Matrix Multiplication using private memory
=================================

Goal
----
* Use private memory to minimize memory movement costs and optimize performance of your matrix multiplication program

Procedure
---------
* Start with the provided matrix multiplication OpenCL code.
* Complete the kernels *C\_row.cl*: each work-item compute a full row of C, and *C\_row\_priv.cl*: row per work-item, A row private.
* In addition, complete the blank parts in the host code *matmul* to run both kernels.

Expected output
---------------
* A message verifying the code has completed successfully.
* Report the runtime and the MFLOPS.
