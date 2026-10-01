cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(dvllm LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_HIP_ARCHITECTURES "gfx1102")
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(ENABLE_EXTRAS OFF CACHE BOOL "Enable extra binaries in extras/ folder")
set(DISABLE_APPLE_METAL ON CACHE BOOL "Disable Apple Metal backend")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_subdirectory(ext/simdcrypt)

include(FetchContent)
FetchContent_Declare(
  Corrosion
  GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
  GIT_TAG v0.6.0)
FetchContent_MakeAvailable(Corrosion)

corrosion_import_crate(MANIFEST_PATH Cargo.toml CRATES mcs-cpp)

# add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()

if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCM has been installed")
    elseif(DEFINED ENV{HIP_PATH})
        set(ROCM_PATH "$ENV{HIP_PATH}/.." CACHE PATH "Path to which ROCM has been installed")
    else()
        set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCM has been installed")
    endif()
endif()

if(NOT DEFINED HCC_PATH)
    if(DEFINED ENV{HCC_PATH})
        set(HCC_PATH $ENV{HCC_PATH} CACHE PATH "Path to which HCC has been installed")
    else()
        set(HCC_PATH "${ROCM_PATH}/hcc" CACHE PATH "Path to which HCC has been installed")
    endif()
    set(HCC_HOME "${HCC_PATH}")
endif()

if(NOT DEFINED HIP_CLANG_PATH)
    if(NOT DEFINED ENV{HIP_CLANG_PATH})
        set(HIP_CLANG_PATH "${ROCM_PATH}/llvm/bin" CACHE PATH "Path to which HIP compatible clang binaries have been installed")
    else()
        set(HIP_CLANG_PATH $ENV{HIP_CLANG_PATH} CACHE PATH "Path to which HIP compatible clang binaries have been installed")
    endif()
endif()

set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_PREFIX_PATH
    "${HIP_PATH}/lib/cmake"
    "${HIP_PATH}/../lib/cmake" # hopefully catches all extra HIP dependencies
)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings")

# find pcre, only needed for client
find_package(PCRE)
if (PCRE_FOUND)
    if (ENABLE_EXTRAS)
        add_executable(llama3.2 
            extras/llama3.2.cpp 
        )
        target_link_libraries(llama3.2  ${PCRE_LIBRARIES})
        target_link_libraries(llama3.2 simdcrypt)
        target_include_directories(llama3.2 PRIVATE ${PCRE_INCLUDE_DIRS})
    endif()

    add_executable(llama3.2q
        src/llama3.2q.cpp
    )
    target_link_libraries(llama3.2q ${PCRE_LIBRARIES})
    target_link_libraries(llama3.2q simdcrypt)
    target_link_libraries(llama3.2q mcscpp)
    target_include_directories(llama3.2q PRIVATE ${PCRE_INCLUDE_DIRS})
endif()

# client file for llama2
if (ENABLE_EXTRAS)
    add_executable(llama2
        extras/llama2.cpp
    )

    target_link_libraries(llama2 simdcrypt)
    target_link_libraries(llama2 mcscpp)
endif()

# client file for llama2
add_executable(llama2q
    src/llama2q.cpp
)

target_link_libraries(llama2q simdcrypt)
target_link_libraries(llama2q mcscpp)

add_executable(quantize
    src/quantize.cpp
)
target_link_libraries(quantize simdcrypt)
target_link_libraries(quantize mcscpp)

# client, quantizer for gemma3
add_executable(gemma3q
    src/gemma3q.cpp
)
target_link_libraries(gemma3q simdcrypt)
target_link_libraries(gemma3q mcscpp)

add_executable(quantize_gemma3
    src/quantize_gemma3.cpp
)
target_link_libraries(quantize_gemma3 simdcrypt)
target_link_libraries(quantize_gemma3 mcscpp)

if (ENABLE_EXTRAS)
    add_executable(server
        extras/server.cpp
    )
endif()

add_executable(serverq
    src/serverq.cpp
)
target_link_libraries(serverq simdcrypt)

add_executable(serverq_gemma3
    src/serverq_gemma3.cpp
)
target_link_libraries(serverq_gemma3 simdcrypt)

if (OPENMP_FOUND)
    if (ENABLE_EXTRAS)
        target_link_libraries(server OpenMP::OpenMP_CXX)
    endif()
    target_link_libraries(serverq OpenMP::OpenMP_CXX)
    target_link_libraries(serverq_gemma3 OpenMP::OpenMP_CXX)
endif()
set_source_files_properties(src/server.cpp PROPERTIES LANGUAGE CXX)
set_source_files_properties(src/serverq.cpp PROPERTIES LANGUAGE CXX)
set_source_files_properties(src/serverq_gemma3.cpp PROPERTIES LANGUAGE CXX)

# find_package(HIP)
# add_executable(matmul
#     matmul.cpp
# )
# set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE HIP)
# target_link_libraries(matmul hip::host)
# ==============================================================================
# APPLE METAL INTEGRATION
# ==============================================================================
if (APPLE AND NOT DISABLE_APPLE_METAL)
    message(STATUS "Apple platform detected. Enabling Metal backend.")
    
    if (ENABLE_EXTRAS)
        target_compile_definitions(server PRIVATE USE_METAL)
    endif()
    target_compile_definitions(serverq PRIVATE USE_METAL)
    target_compile_definitions(serverq_gemma3 PRIVATE USE_METAL)

    # Force Apple Clang to treat the core files as Objective-C++ and enable Automatic Reference Counting
    set_source_files_properties(src/server.cpp src/serverq.cpp src/serverq_gemma3.cpp PROPERTIES
        COMPILE_FLAGS "-x objective-c++ -fobjc-arc"
    )

    # Link the native macOS frameworks required for the GPU and shader pipelines
    if (ENABLE_EXTRAS)
        target_link_libraries(server "-framework Foundation" "-framework Metal" "-framework MetalPerformanceShaders")
        target_link_libraries(server "-framework Accelerate")
    endif()
    target_link_libraries(serverq "-framework Foundation" "-framework Metal" "-framework MetalPerformanceShaders")
    target_link_libraries(serverq "-framework Accelerate")
    target_link_libraries(serverq_gemma3 "-framework Foundation" "-framework Metal" "-framework MetalPerformanceShaders")
    target_link_libraries(serverq_gemma3 "-framework Accelerate")
endif()
# ==============================================================================

find_package(rocblas)

if (rocblas_FOUND)
    enable_language(HIP)
    if (ENABLE_EXTRAS)
        target_compile_definitions(server PRIVATE USE_ROCBLAS)
    endif()
    target_compile_definitions(serverq PRIVATE USE_ROCBLAS)
    target_compile_definitions(serverq_gemma3 PRIVATE USE_ROCBLAS)

    # kernel for int32 matmul
    add_library(rocblas_int32_matmul STATIC src/rocblas_int32_matmul.hip)
    set_source_files_properties(src/rocblas_int32_matmul.hip PROPERTIES LANGUAGE HIP)
    target_link_libraries(rocblas_int32_matmul hip::host)

    # rocblas_int32_matmul is unnecesary for server, but meh
    if (ENABLE_EXTRAS)
        add_dependencies(server rocblas_int32_matmul)
        target_link_libraries(server hip::host roc::rocblas rocblas_int32_matmul)
    endif()

    add_dependencies(serverq rocblas_int32_matmul)
    target_link_libraries(serverq hip::host roc::rocblas rocblas_int32_matmul)

    add_dependencies(serverq_gemma3 rocblas_int32_matmul)
    target_link_libraries(serverq_gemma3 hip::host roc::rocblas rocblas_int32_matmul)
endif()

find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    # add_compile_options(
    #     $<$<COMPILE_LANGUAGE:CUDA>:
    #         -O3 
    #         # --use_fast_math 
    #         # -Xptxas=-O3 
    #         # -Xcompiler=-O3,-march=native
    #     >
    # )
    if (ENABLE_EXTRAS)
        target_compile_definitions(server PRIVATE USE_CUBLAS)
    endif()
    target_compile_definitions(serverq PRIVATE USE_CUBLAS)
    target_compile_definitions(serverq_gemma3 PRIVATE USE_CUBLAS)

    # kernel for int matmul
    add_library(cuda_int32_matmul STATIC src/cuda_int32_matmul.cu)
    target_compile_options(cuda_int32_matmul PRIVATE
        # Sent only to GCC/Clang host compiler
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
        
        # Sent to NVCC (and safely forwarded to host compiler if needed)
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native>
    )
    set_source_files_properties(src/cuda_int32_matmul.cu PROPERTIES LANGUAGE CUDA)
    target_link_libraries(cuda_int32_matmul CUDA::cudart CUDA::cublas CUDA::curand)

    if (ENABLE_EXTRAS)
        add_dependencies(server cuda_int32_matmul)
        target_link_libraries(server cuda_int32_matmul)
    endif()

    add_dependencies(serverq cuda_int32_matmul)
    target_link_libraries(serverq cuda_int32_matmul)

    add_dependencies(serverq_gemma3 cuda_int32_matmul)
    target_link_libraries(serverq_gemma3 cuda_int32_matmul)
endif()
