# Deeper - Compile time shape checked GPU accelerated linear algebra

Deeper is a GPU accelerated linear algebra library for Rust. It extends up on the ideas introduced by the nalgebra library and adds compile time shape checking to the linear algebra operations. 
This allows the compiler to catch errors at compile time instead of runtime. 
Deeper also adds GPU acceleration to the linear algebra operations.
Deeper is currently in a very very early experimental stage of development and is not ready for production use.
It relies on typenum and cust for compile time shape checking and GPU acceleration.

## Build

First build the `cuda-build` project, it will create the ptx files needed for the GPU acceleration.
Then build the `deeper` project.
