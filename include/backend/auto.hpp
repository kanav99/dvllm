#ifdef USE_ROCBLAS
    #include "backend/rocblas.hpp"
    using AutoBackend = RocblasBackend;
#elif USE_CUBLAS
    #include "backend/cublas.hpp"
    using AutoBackend = CublasBackend;
#elif USE_METAL
    #include "backend/metal.hpp"
    using AutoBackend = MetalBackend;
#else
    // no backend, so we will use the CPU implementation
    #include "backend/cpu.hpp"
    using AutoBackend = CpuBackend;
#endif
