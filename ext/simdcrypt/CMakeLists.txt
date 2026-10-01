cmake_minimum_required(VERSION 3.16)
project(simdcrypt)

enable_testing()
set(CMAKE_CXX_STANDARD 20)

if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64" OR "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "AMD64")
    # if Intel machine, use `-maes` flag
    message(STATUS "${PROJECT_NAME}: Using Intel AES-NI")
    set(SIMDCRYPT_CXX_FLAGS "-maes" "-march=native")
elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64" OR "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64")
    # if ARM machine, use `-march=armv8-a+crypto` flag
    message(STATUS "${PROJECT_NAME}: Using ARM Neon")
    set(SIMDCRYPT_CXX_FLAGS "-march=armv8-a+crypto")
endif()

add_library(${PROJECT_NAME} STATIC
    src/AES.cpp
    src/PRNG.cpp
)

# set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SIMDCRYPT_CXX_FLAGS}")
target_compile_options(${PROJECT_NAME} PUBLIC ${SIMDCRYPT_CXX_FLAGS})

target_include_directories(${PROJECT_NAME}
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

add_executable(basic-test tests/basic.cpp)
target_link_libraries(basic-test PRIVATE ${PROJECT_NAME})
add_test(NAME basic-test COMMAND basic-test)

add_executable(block-test tests/block.cpp)
target_link_libraries(block-test PRIVATE ${PROJECT_NAME})
add_test(NAME block-test COMMAND block-test)

add_executable(prng-test tests/prng.cpp)
target_link_libraries(prng-test PRIVATE ${PROJECT_NAME})
add_test(NAME prng-test COMMAND prng-test)

add_executable(hash-test tests/hash.cpp)
target_link_libraries(hash-test PRIVATE ${PROJECT_NAME})
add_test(NAME hash-test COMMAND hash-test)

add_executable(sha256-test tests/sha256.cpp)
target_link_libraries(sha256-test PRIVATE ${PROJECT_NAME})
add_test(NAME sha256-test COMMAND sha256-test)

find_package(OpenSSL)

if (OpenSSL_FOUND)
    message(STATUS "${PROJECT_NAME}: Found OpenSSL, enabling OpenSSL tests")
    add_executable(openssl-test tests/openssl.cpp)
    target_link_libraries(openssl-test PRIVATE ${PROJECT_NAME} OpenSSL::Crypto)
    add_test(NAME openssl-test COMMAND openssl-test)
else()
    message(WARNING "${PROJECT_NAME}: OpenSSL not found, skipping OpenSSL tests")
    return()
endif()
