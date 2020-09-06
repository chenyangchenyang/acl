#pragma once

////////////////////////////////////////////////////////////////////////////////
// The MIT License (MIT)
//
// Copyright (c) 2020 Nicholas Frechette & Animation Compression Library contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
////////////////////////////////////////////////////////////////////////////////

#include "acl/core/impl/compiler_utils.h"

#include <cstdint>

ACL_IMPL_FILE_PRAGMA_PUSH

namespace acl
{
	// TODO: Add support for streaming prefetch (ptr, 0, 0) for arm
	inline void memory_prefetch(const void* ptr)
	{
#if defined(RTM_SSE2_INTRINSICS)
		_mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
#elif defined(ACL_COMPILER_GCC) || defined(ACL_COMPILER_CLANG)
		__builtin_prefetch(ptr, 0, 3);
#elif defined(RTM_NEON_INTRINSICS) && defined(ACL_COMPILER_MSVC)
		__prefetch(ptr);
#else
		(void)ptr;
#endif
	}

	namespace acl_impl
	{
		template<typename cached_type>
		struct track_cache_v0
		{
			// Our cached values
			cached_type		cached_samples[8];

			// The index to write the next cache entry when we unpack
			// Effective index value is modulo 8 what is stored here, guaranteed to never wrap
			uint32_t		cache_write_index = 0;

			// The index to read the next cache entry when we consume
			// Effective index value is modulo 8 what is stored here, guaranteed to never wrap
			uint32_t		cache_read_index = 0;

			// How many we have left to unpack in total
			uint32_t		num_left_to_unpack;

			// Returns the number of cached entries
			uint32_t		get_num_cached() const { return cache_write_index - cache_read_index; }
		};
	}
}
ACL_IMPL_FILE_PRAGMA_POP
