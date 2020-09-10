#pragma once

////////////////////////////////////////////////////////////////////////////////
// The MIT License (MIT)
//
// Copyright (c) 2019 Nicholas Frechette & Animation Compression Library contributors
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

#include "acl/core/compressed_tracks.h"
#include "acl/core/error_result.h"
#include "acl/core/iallocator.h"
#include "acl/core/impl/compiler_utils.h"
#include "acl/compression/compression_settings.h"
#include "acl/compression/output_stats.h"
#include "acl/compression/track_array.h"

#include <cstdint>

ACL_IMPL_FILE_PRAGMA_PUSH

namespace acl
{
	//////////////////////////////////////////////////////////////////////////
	// Compresses a track array with uniform sampling.
	//
	// This compression algorithm is the simplest by far and as such it offers
	// the fastest compression and decompression. Every sample is retained and
	// every track has the same number of samples playing back at the same
	// sample rate. This means that when we sample at a particular time within
	// the clip, we can trivially calculate the offsets required to read the
	// desired data. All the data is sorted in order to ensure all reads are
	// as contiguous as possible for optimal cache locality during decompression.
	//
	//    allocator:				The allocator instance to use to allocate and free memory.
	//    track_list:				The track list to compress.
	//    settings:					The compression settings to use.
	//    out_compressed_tracks:	The resulting compressed tracks. The caller owns the returned memory and must free it.
	//    out_stats:				Stat output structure.
	//////////////////////////////////////////////////////////////////////////
	error_result compress_track_list(iallocator& allocator, const track_array& track_list, const compression_settings& settings,
		compressed_tracks*& out_compressed_tracks, output_stats& out_stats);

	//////////////////////////////////////////////////////////////////////////
	// Compresses a transform track array using its additive base with uniform sampling.
	//
	// This compression algorithm is the simplest by far and as such it offers
	// the fastest compression and decompression. Every sample is retained and
	// every track has the same number of samples playing back at the same
	// sample rate. This means that when we sample at a particular time within
	// the clip, we can trivially calculate the offsets required to read the
	// desired data. All the data is sorted in order to ensure all reads are
	// as contiguous as possible for optimal cache locality during decompression.
	//
	//    allocator:				The allocator instance to use to allocate and free memory.
	//    track_list:				The track list to compress.
	//    settings:					The compression settings to use.
	//    out_compressed_tracks:	The resulting compressed tracks. The caller owns the returned memory and must free it.
	//    out_stats:				Stat output structure.
	//////////////////////////////////////////////////////////////////////////
	error_result compress_track_list(iallocator& allocator, const track_array_qvvf& track_list, const compression_settings& settings,
		const track_array_qvvf& additive_base_track_list, additive_clip_format8 additive_format,
		compressed_tracks*& out_compressed_tracks, output_stats& out_stats);

	//////////////////////////////////////////////////////////////////////////
	// Compresses a transform track array using its additive base with uniform sampling.
	//
	// This compression algorithm is the simplest by far and as such it offers
	// the fastest compression and decompression. Every sample is retained and
	// every track has the same number of samples playing back at the same
	// sample rate. This means that when we sample at a particular time within
	// the clip, we can trivially calculate the offsets required to read the
	// desired data. All the data is sorted in order to ensure all reads are
	// as contiguous as possible for optimal cache locality during decompression.
	//
	// The data is also partitioned into a separate database. The resulting compressed clip
	// can be used for playback on its own but with reduced quality and together with the database
	// full quality playback can be achieved. In a separate step, multiple databases can be merged
	// together for bulk streaming. The two step process allows easy preview and a predictable
	// cost when compressing offline.
	//
	//    allocator:				The allocator instance to use to allocate and free memory.
	//    track_list:				The track list to compress.
	//    settings:					The compression settings to use.
	//    out_compressed_tracks:	The resulting compressed tracks. The caller owns the returned memory and must free it.
	//    out_compressed_database:	The resulting compressed database. The caller owns the returned memory and must free it.
	//    out_stats:				Stat output structure.
	//////////////////////////////////////////////////////////////////////////
	error_result compress_track_list(iallocator& allocator, const track_array_qvvf& track_list, const compression_settings& settings,
		const track_array_qvvf& additive_base_track_list, additive_clip_format8 additive_format,
		compressed_tracks*& out_compressed_tracks, compressed_database*& out_compressed_database, output_stats& out_stats);

	//////////////////////////////////////////////////////////////////////////
	// A pair of pointers to a compressed tracks instance and its database.
	//////////////////////////////////////////////////////////////////////////
	struct database_merge_mapping
	{
		// The compressed tracks and its associated compressed database
		compressed_tracks* tracks;
		const compressed_database* database;

		// TODO: Implement is_valid function that checks the mapping is valid
	};

	//////////////////////////////////////////////////////////////////////////
	// Merges the provided compressed databases together into a new instance.
	// The input databases are left unchanged (read only) and will no longer be
	// referenced by the compressed_tracks instances provided.
	// The input compressed_tracks instances will be updated to point to the new merged
	// database.
	//
	//    allocator:						The allocator instance to use to allocate the new database.
	//    merge_mappings:					The mappings to merge together into our new database.
	//    num_merge_mappings:				The number of mappings to merge together.
	//    out_merged_compressed_database:	The resulting merged database. The caller owns the returned memory and must free it.
	//////////////////////////////////////////////////////////////////////////
	error_result merge_compressed_databases(iallocator& allocator, const database_merge_mapping* merge_mappings, uint32_t num_merge_mappings, compressed_database*& out_merged_compressed_database);
}

#include "acl/compression/impl/compress.impl.h"
#include "acl/compression/impl/compress.database.impl.h"

ACL_IMPL_FILE_PRAGMA_POP
