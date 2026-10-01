// backend.hpp

#pragma once

#include "constants.hpp"

class Backend
{
public:
    virtual ~Backend() = default;

    // Multi-GPU context management
    virtual void initDevices(int num_devices) {}
    virtual void setDevice(int device) {}
    virtual int getNumDevices() { return 1; }

    // Perform matrix multiplication
    virtual void matmul(float *d_y, float *d_x, float *d_w, int n, int d) = 0;
    virtual void matmul(integer_t *d_y, integer_t *d_x, integer_w_t *d_w, int n, int d, int batch_size = 1) = 0;

    virtual void malloc(void **d_ptr, size_t size) = 0;
    void malloc(float **d_ptr, size_t size) { malloc((void **)d_ptr, size); }
    void malloc(integer_t **d_ptr, size_t size) { malloc((void **)d_ptr, size); }

    // enable only if integet_t and integer_w_t are different types
    template<typename = void> 
    requires (!std::same_as<integer_t, integer_w_t>)
    void malloc(integer_w_t **d_ptr, size_t size) { malloc((void **)d_ptr, size); }


    virtual void free(void *d_ptr) = 0;
    void free(float *d_ptr) { free((void *)d_ptr); }
    void free(integer_t *d_ptr) { free((void *)d_ptr); }

    // enable only if integet_t and integer_w_t are different types
    template<typename = void>
    requires (!std::same_as<integer_t, integer_w_t>)
    void free(integer_w_t *d_ptr) { free((void *)d_ptr); }

    virtual void copyHostToDevice(void *d_ptr, void *h_ptr, size_t size) = 0;
    void copyHostToDevice(float *d_ptr, float *h_ptr, size_t size) { copyHostToDevice((void *)d_ptr, (void *)h_ptr, size); }
    void copyHostToDevice(integer_t *d_ptr, integer_t *h_ptr, size_t size) { copyHostToDevice((void *)d_ptr, (void *)h_ptr, size); }

    virtual void copyDeviceToHost(void *h_ptr, void *d_ptr, size_t size) = 0;
    void copyDeviceToHost(float *h_ptr, float *d_ptr, size_t size) { copyDeviceToHost((void *)h_ptr, (void *)d_ptr, size); }
    void copyDeviceToHost(integer_t *h_ptr, integer_t *d_ptr, size_t size) { copyDeviceToHost((void *)h_ptr, (void *)d_ptr, size); }

    // virtual void quantizeHostToDevice(integer_w_t *d_ptr, float *h_ptr, size_t num_elements) = 0;

    // async copyHostToDevice
    virtual void copyHostToDeviceAsync(void *d_ptr, void *h_ptr, size_t size, int handle) {
        copyHostToDevice(d_ptr, h_ptr, size); // synchronous fallback
    }
    virtual void copyHostToDeviceAsync(float *d_ptr, float *h_ptr, size_t size, int handle) {
        return copyHostToDeviceAsync((void *)d_ptr, (void *)h_ptr, size, handle);
    }
    virtual void copyHostToDeviceAsync(integer_t *d_ptr, integer_t *h_ptr, size_t size, int handle) {
        return copyHostToDeviceAsync((void *)d_ptr, (void *)h_ptr, size, handle);
    }
    virtual void waitForOperation(int handle) {
        return;
    }

    virtual void pinMemory(void *ptr, size_t size) {}
    virtual void unpinMemory(void *ptr) {}

    virtual bool isDeviceAvailable() = 0;
    virtual const char* getDeviceName() = 0;

    virtual bool isCpuBackend() { return false; }
};
