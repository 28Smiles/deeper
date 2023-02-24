mod tensor;
mod shape;

thread_local! {
    pub(crate) static STREAM: cust::stream::Stream = cust::stream::Stream::new(cust::stream::StreamFlags::NON_BLOCKING, None).unwrap();
    pub(crate) static CTX: cust::context::Context = cust::quick_init().unwrap();
    pub(crate) static MODULE: cust::module::Module = cust::module::Module::from_ptx(include_str!("../resources/cuda.ptx"), &[]).unwrap();
}
