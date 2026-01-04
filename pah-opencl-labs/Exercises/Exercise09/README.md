Exercise 9 - Heterogeneous Computing
====================================

Goal
----
* To experiment with running kernels on multiple devices.

Procedure
---------
* Take the vadd OpenCL program (Exercise 2).
* Investigate the Context constructors and include more than once device.
* Modify the program to run a kernel on multiple devices, each with different input data.
* Split your problem across multiple devices.
* Use the examples from the NVIDIA OpenCL SDKs to help you.

Expected output
---------------
* A message verifying that the vector addition completed successfully.
* See which device runs faster.
