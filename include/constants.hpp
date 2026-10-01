#pragma once

#include <cstdint>

using integer_t = int64_t;
using integer_w_t = int16_t;
const float FP_SCALE = 2048; // scale factor for quantization
const float FP_W_SCALE = 2048; // scale factor for quantization of weights
const int CHUNK_SIZE = 2048;
