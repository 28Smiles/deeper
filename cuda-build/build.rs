use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../cuda")
        .copy_to("./../resources/cuda.ptx")
        .build()
        .unwrap();
}
