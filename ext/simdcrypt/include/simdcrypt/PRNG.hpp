#pragma once
// This file and the associated implementation has been placed in the public domain, waiving all copyright. No restrictions are placed on its use.
#include "AES.hpp"
#include <span>
#include <vector>

namespace simdcrypt
{

	// A Peudorandom number generator implemented using AES-NI.
    class PRNG
    {
    public:

		// default construct leaves the PRNG in an invalid state.
		// SetSeed(...) must be called before get(...)
        PRNG() = default;

		// explicit constructor to initialize the PRNG with the 
		// given seed and to buffer bufferSize number of AES block
        PRNG(const block& seed, uint64_t bufferSize = 256);

		// standard move constructor. The moved from PRNG is invalid
		// unless SetSeed(...) is called.
        PRNG(PRNG&& s);

		// Copy is not allowed.
        PRNG(const PRNG&) = delete;

        // standard move assignment. The moved from PRNG is invalid
        // unless SetSeed(...) is called.
        void operator=(PRNG&&);

        // Set seed from a block and set the desired buffer size.
        void SetSeed(const block& b, uint64_t bufferSize = 256);

		// Return the seed for this PRNG.
        const block getSeed() const;


        struct AnyPOD
        {
            PRNG& mPrng;

            template<typename T, typename U = typename std::enable_if<std::is_standard_layout<T>::value, T>::type>
                operator T()
            {
                return mPrng.get<T>();
            }

        };

        AnyPOD get()
        {
            return { *this };
        }

		// Templated function that returns the a random element
		// of the given type T. 
		// Required: T must be a standard layout type.
        template<typename T>
        typename std::enable_if<std::is_standard_layout<T>::value, T>::type
			get()
        {
            T ret;
            get((uint8_t*)&ret, sizeof(T));
            return ret;
        }

		// Templated function that fills the provided buffer 
		// with random elements of the given type T. 
		// Required: T must be a standard layout type.
        template<typename T>
		typename std::enable_if<std::is_standard_layout<T>::value, void>::type 
			get(T* dest, uint64_t length)
        {
            uint64_t lengthuint8_t = length * sizeof(T);
            uint8_t* destuint8_t = (uint8_t*)dest;
            while (lengthuint8_t)
            {
                uint64_t step = std::min(lengthuint8_t, mBufferByteCapacity - mBytesIdx);

                memcpy(destuint8_t, ((uint8_t*)mBuffer.data()) + mBytesIdx, step);

                destuint8_t += step;
                lengthuint8_t -= step;
                mBytesIdx += step;

                if (mBytesIdx == mBufferByteCapacity)
                    refillBuffer();
            }
        }

		// Templated function that fills the provided buffer 
		// with random elements of the given type T. 
		// Required: T must be a POD type.
		template<typename T>
		typename std::enable_if<std::is_standard_layout<T>::value, void>::type
			get(std::span<T> dest)
		{
			get(dest.data(), dest.size());
		}

        // returns the buffer of maximum maxSize bytes or however 
        // many the internal buffer has, which ever is smaller. The 
        // returned bytes are "consumed" and will not be used on 
        // later calls to get*(...). Note, buffer may be invalidated 
        // on the next call to get*(...) or destruction.
        std::span<uint8_t> getBufferSpan(uint64_t maxSize)
        {
            if (mBytesIdx == mBufferByteCapacity)
                refillBuffer();

            auto data = ((uint8_t*)mBuffer.data()) + mBytesIdx;
            auto size = std::min(maxSize, mBufferByteCapacity - mBytesIdx);

            mBytesIdx += size;

            return std::span<uint8_t>(data, size);
        }

		// Returns a random element from {0,1}
        uint8_t getBit();

		// STL random number interface
        typedef uint64_t result_type;
        static constexpr result_type min() { return 0; }
        static constexpr result_type max() { return (result_type)-1; }
        result_type operator()() {
            return get<result_type>();
        }

        template<typename R>
        R operator()(R mod) {
            return get<typename std::make_unsigned<R>::type>() % mod;
        }

		// internal buffer to store future random values.
		std::vector<block> mBuffer;

		// AES that generates the randomness by computing AES_seed({0,1,2,...})
		AES mAes;

		// Indicators denoting the current state of the buffer.
		uint64_t mBytesIdx = 0,
			mBlockIdx = 0,
			mBufferByteCapacity = 0;

		// refills the internal buffer with fresh randomness
		void refillBuffer();
    };

	// specialization to make bool work correctly.
    template<>
    inline void PRNG::get<bool>(bool* dest, uint64_t length)
    {
        get((uint8_t*)dest, length);
        for (uint64_t i = 0; i < length; ++i) dest[i] = ((uint8_t*)dest)[i] & 1;
    }

	// specialization to make bool work correctly.
    template<>
    inline bool PRNG::get<bool>()
    {
        uint8_t ret;
        get((uint8_t*)&ret, 1);
        return ret & 1;
    }


	template<typename T>
	typename std::enable_if<std::is_standard_layout<T>::value, PRNG&>::type operator<<(T& rhs, PRNG& lhs)
	{
		lhs.get(&rhs, 1);
	}

}
