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

#include "acl/core/bitset.h"
#include "acl/core/compressed_tracks.h"
#include "acl/core/compressed_tracks_version.h"
#include "acl/core/interpolation_utils.h"
#include "acl/core/range_reduction_types.h"
#include "acl/core/track_formats.h"
#include "acl/core/track_writer.h"
#include "acl/core/variable_bit_rates.h"
#include "acl/core/impl/compiler_utils.h"
#include "acl/math/quatf.h"
#include "acl/math/quat_packing.h"
#include "acl/math/vector4f.h"

#include <rtm/scalarf.h>
#include <rtm/vector4f.h>

#include <cstdint>
#include <type_traits>

#define ACL_IMPL_USE_CONSTANT_PREFETCH
#define ACL_IMPL_USE_ANIMATED_PREFETCH
//#define ACL_IMPL_VEC3_UNPACK

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
#endif
	}

#if defined(ACL_IMPL_USE_CONSTANT_PREFETCH)
	#define ACL_IMPL_CONSTANT_PREFETCH(ptr) memory_prefetch(ptr)
#else
	#define ACL_IMPL_CONSTANT_PREFETCH(ptr) (void)ptr
#endif

#if defined(ACL_IMPL_USE_ANIMATED_PREFETCH)
	#define ACL_IMPL_ANIMATED_PREFETCH(ptr) memory_prefetch(ptr)
#else
	#define ACL_IMPL_ANIMATED_PREFETCH(ptr) (void)ptr
#endif

	namespace acl_impl
	{
		struct alignas(64) persistent_transform_decompression_context_v0
		{
			// Clip related data							//   offsets
			// Only member used to detect if we are initialized, must be first
			const compressed_tracks* tracks;				//   0 |   0

			const uint32_t* constant_tracks_bitset;			//   4 |   8
			const uint8_t* constant_track_data;				//   8 |  16
			const uint32_t* default_tracks_bitset;			//  12 |  24

			const uint8_t* clip_range_data;					//  16 |  32

			float clip_duration;							//  20 |  40

			bitset_description bitset_desc;					//  24 |  44

			uint32_t clip_hash;								//  28 |  48

			rotation_format8 rotation_format;				//  32 |  52
			vector_format8 translation_format;				//  33 |  53
			vector_format8 scale_format;					//  34 |  54
			range_reduction_flags8 range_reduction;			//  35 |  55

			uint8_t num_rotation_components;				//  36 |  56
			uint8_t has_segments;							//  37 |  57

			uint8_t padding0[2];							//  38 |  58

			// Seeking related data
			float sample_time;								//  40 |  60

			const uint8_t* format_per_track_data[2];		//  44 |  64
			const uint8_t* segment_range_data[2];			//  52 |  80
			const uint8_t* animated_track_data[2];			//  60 |  96

			uint32_t key_frame_bit_offsets[2];				//  68 | 112

			float interpolation_alpha;						//  76 | 120

			uint8_t padding1[sizeof(void*) == 4 ? 48 : 4];	//  80 | 124

			//									Total size:	   128 | 128

			//////////////////////////////////////////////////////////////////////////

			const compressed_tracks* get_compressed_tracks() const { return tracks; }
			compressed_tracks_version16 get_version() const { return tracks->get_version(); }
			bool is_initialized() const { return tracks != nullptr; }
			void reset() { tracks = nullptr; }
		};

		static_assert(sizeof(persistent_transform_decompression_context_v0) == 128, "Unexpected size");

		struct alignas(64) sampling_context_v0
		{
			//														//   offsets
			uint32_t track_index;									//   0 |   0
			uint32_t constant_track_data_offset;					//   4 |   4
			uint32_t clip_range_data_offset;						//   8 |   8

			uint32_t format_per_track_data_offset;					//  12 |  12
			uint32_t segment_range_data_offset;						//  16 |  16

			uint32_t key_frame_bit_offsets[2];						//  20 |  20

			uint8_t padding[4];										//  28 |  28

			rtm::vector4f vectors[2];								//  32 |  32

			//											Total size:	    64 |  64
		};

		static_assert(sizeof(sampling_context_v0) == 64, "Unexpected size");

		// We use adapters to wrap the decompression_settings
		// This allows us to re-use the code for skipping and decompressing Vector3 samples
		// Code generation will generate specialized code for each specialization
		template<class decompression_settings_type>
		struct translation_decompression_settings_adapter
		{
			// Forward to our decompression settings
			static constexpr range_reduction_flags8 get_range_reduction_flag() { return range_reduction_flags8::translations; }
			static constexpr vector_format8 get_vector_format(const persistent_transform_decompression_context_v0& context) { return context.translation_format; }
			static constexpr bool is_vector_format_supported(vector_format8 format) { return decompression_settings_type::is_translation_format_supported(format); }
		};

		template<class decompression_settings_type>
		struct scale_decompression_settings_adapter
		{
			// Forward to our decompression settings
			static constexpr range_reduction_flags8 get_range_reduction_flag() { return range_reduction_flags8::scales; }
			static constexpr vector_format8 get_vector_format(const persistent_transform_decompression_context_v0& context) { return context.scale_format; }
			static constexpr bool is_vector_format_supported(vector_format8 format) { return decompression_settings_type::is_scale_format_supported(format); }
		};

		// Returns the statically known number of rotation formats supported by the decompression settings
		template<class decompression_settings_type>
		constexpr int32_t num_supported_rotation_formats()
		{
			return decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_full)
				+ decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_drop_w_full)
				+ decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_drop_w_variable);
		}

		// Returns the statically known rotation format supported if we only support one, otherwise we return the input value
		// which might not be known statically
		template<class decompression_settings_type>
		constexpr rotation_format8 get_rotation_format(rotation_format8 format)
		{
			return num_supported_rotation_formats<decompression_settings_type>() > 1
				// More than one format is supported, return the input value, whatever it may be
				? format
				// Only one format is supported, figure out statically which one it is and return it
				: (decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_full) ? rotation_format8::quatf_full
					: (decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_drop_w_full) ? rotation_format8::quatf_drop_w_full
						: rotation_format8::quatf_drop_w_variable));
		}

		// Returns the statically known number of vector formats supported by the decompression settings
		template<class decompression_settings_adapter_type>
		constexpr int32_t num_supported_vector_formats()
		{
			return decompression_settings_adapter_type::is_vector_format_supported(vector_format8::vector3f_full)
				+ decompression_settings_adapter_type::is_vector_format_supported(vector_format8::vector3f_variable);
		}

		// Returns the statically known vector format supported if we only support one, otherwise we return the input value
		// which might not be known statically
		template<class decompression_settings_adapter_type>
		constexpr vector_format8 get_vector_format(vector_format8 format)
		{
			return num_supported_vector_formats<decompression_settings_adapter_type>() > 1
				// More than one format is supported, return the input value, whatever it may be
				? format
				// Only one format is supported, figure out statically which one it is and return it
				: (decompression_settings_adapter_type::is_vector_format_supported(vector_format8::vector3f_full) ? vector_format8::vector3f_full
					: vector_format8::vector3f_variable);
		}

		template<class decompression_settings_type>
		inline void skip_over_rotation(const persistent_transform_decompression_context_v0& decomp_context, sampling_context_v0& sampling_context_)
		{
			const bitset_index_ref track_index_bit_ref(decomp_context.bitset_desc, sampling_context_.track_index);
			const bool is_sample_default = bitset_test(decomp_context.default_tracks_bitset, track_index_bit_ref);
			if (!is_sample_default)
			{
				const rotation_format8 rotation_format = get_rotation_format<decompression_settings_type>(decomp_context.rotation_format);

				const bool is_sample_constant = bitset_test(decomp_context.constant_tracks_bitset, track_index_bit_ref);
				if (is_sample_constant)
				{
					const rotation_format8 packed_format = is_rotation_format_variable(rotation_format) ? get_highest_variant_precision(get_rotation_variant(rotation_format)) : rotation_format;
					sampling_context_.constant_track_data_offset += get_packed_rotation_size(packed_format);
				}
				else
				{
					if (is_rotation_format_variable(rotation_format))
					{
						for (uint32_t i = 0; i < 2; ++i)
						{
							const uint8_t bit_rate = decomp_context.format_per_track_data[i][sampling_context_.format_per_track_data_offset];
							const uint32_t num_bits_at_bit_rate = get_num_bits_at_bit_rate(bit_rate) * 3;	// 3 components

							sampling_context_.key_frame_bit_offsets[i] += num_bits_at_bit_rate;
						}

						sampling_context_.format_per_track_data_offset++;
					}
					else
					{
						const uint32_t rotation_size = get_packed_rotation_size(rotation_format);
						const uint32_t num_bits_at_bit_rate = rotation_size == (sizeof(float) * 4) ? 128 : 96;

						for (uint32_t i = 0; i < 2; ++i)
							sampling_context_.key_frame_bit_offsets[i] += num_bits_at_bit_rate;
					}

					if (are_any_enum_flags_set(decomp_context.range_reduction, range_reduction_flags8::rotations))
					{
						sampling_context_.clip_range_data_offset += decomp_context.num_rotation_components * sizeof(float) * 2;

						if (decomp_context.has_segments)
							sampling_context_.segment_range_data_offset += decomp_context.num_rotation_components * k_segment_range_reduction_num_bytes_per_component * 2;
					}
				}
			}

			sampling_context_.track_index++;
		}

		template <class decompression_settings_type>
		inline rtm::quatf RTM_SIMD_CALL decompress_and_interpolate_rotation(const persistent_transform_decompression_context_v0& decomp_context, sampling_context_v0& sampling_context_)
		{
			rtm::quatf interpolated_rotation;

			const bitset_index_ref track_index_bit_ref(decomp_context.bitset_desc, sampling_context_.track_index);
			const bool is_sample_default = bitset_test(decomp_context.default_tracks_bitset, track_index_bit_ref);
			if (is_sample_default)
			{
				interpolated_rotation = rtm::quat_identity();
			}
			else
			{
				const rotation_format8 rotation_format = get_rotation_format<decompression_settings_type>(decomp_context.rotation_format);
				const bool is_sample_constant = bitset_test(decomp_context.constant_tracks_bitset, track_index_bit_ref);
				if (is_sample_constant)
				{
					if (rotation_format == rotation_format8::quatf_full && decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_full))
						interpolated_rotation = unpack_quat_128(decomp_context.constant_track_data + sampling_context_.constant_track_data_offset);
					else if (rotation_format == rotation_format8::quatf_drop_w_full && decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_drop_w_full))
						interpolated_rotation = unpack_quat_96_unsafe(decomp_context.constant_track_data + sampling_context_.constant_track_data_offset);
					else if (rotation_format == rotation_format8::quatf_drop_w_variable && decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_drop_w_variable))
						interpolated_rotation = unpack_quat_96_unsafe(decomp_context.constant_track_data + sampling_context_.constant_track_data_offset);
					else
					{
						ACL_ASSERT(false, "Unrecognized rotation format");
						interpolated_rotation = rtm::quat_identity();
					}

					ACL_ASSERT(rtm::quat_is_finite(interpolated_rotation), "Rotation is not valid!");
					ACL_ASSERT(rtm::quat_is_normalized(interpolated_rotation), "Rotation is not normalized!");

					const rotation_format8 packed_format = is_rotation_format_variable(rotation_format) ? get_highest_variant_precision(get_rotation_variant(rotation_format)) : rotation_format;
					sampling_context_.constant_track_data_offset += get_packed_rotation_size(packed_format);
				}
				else
				{
					// This part is fairly complex, we'll loop and write to the stack (sampling context)
					rtm::vector4f* rotations_as_vec = &sampling_context_.vectors[0];

					// Range ignore flags are used to skip range normalization at the clip and/or segment levels
					// Each sample has two bits like so:
					//    - 0x01 = sample 1 segment
					//    - 0x02 = sample 1 clip
					//    - 0x04 = sample 0 segment
					//    - 0x08 = sample 0 clip
					// By default, we never ignore range reduction
					uint32_t range_ignore_flags = 0;

					if (rotation_format == rotation_format8::quatf_drop_w_variable && decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_drop_w_variable))
					{
						for (uint32_t i = 0; i < 2; ++i)
						{
							range_ignore_flags <<= 2;

							const uint8_t bit_rate = decomp_context.format_per_track_data[i][sampling_context_.format_per_track_data_offset];
							const uint32_t num_bits_at_bit_rate = get_num_bits_at_bit_rate(bit_rate);

							if (is_constant_bit_rate(bit_rate))
							{
								rotations_as_vec[i] = unpack_vector3_u48_unsafe(decomp_context.segment_range_data[i] + sampling_context_.segment_range_data_offset);
								range_ignore_flags |= 0x00000001U;	// Skip segment only
							}
							else if (is_raw_bit_rate(bit_rate))
							{
								rotations_as_vec[i] = unpack_vector3_96_unsafe(decomp_context.animated_track_data[i], sampling_context_.key_frame_bit_offsets[i]);
								range_ignore_flags |= 0x00000003U;	// Skip clip and segment
							}
							else
								rotations_as_vec[i] = unpack_vector3_uXX_unsafe(uint8_t(num_bits_at_bit_rate), decomp_context.animated_track_data[i], sampling_context_.key_frame_bit_offsets[i]);

							sampling_context_.key_frame_bit_offsets[i] += num_bits_at_bit_rate * 3;
						}

						sampling_context_.format_per_track_data_offset++;
					}
					else
					{
						if (rotation_format == rotation_format8::quatf_full && decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_full))
						{
							for (uint32_t i = 0; i < 2; ++i)
							{
								rotations_as_vec[i] = unpack_vector4_128_unsafe(decomp_context.animated_track_data[i], sampling_context_.key_frame_bit_offsets[i]);
								sampling_context_.key_frame_bit_offsets[i] += 128;
							}
						}
						else if (rotation_format == rotation_format8::quatf_drop_w_full && decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_drop_w_full))
						{
							for (uint32_t i = 0; i < 2; ++i)
							{
								rotations_as_vec[i] = unpack_vector3_96_unsafe(decomp_context.animated_track_data[i], sampling_context_.key_frame_bit_offsets[i]);
								sampling_context_.key_frame_bit_offsets[i] += 96;
							}
						}
					}

					// Load our samples to avoid working with the stack now that things can be unrolled.
					// We unroll because even if we work from the stack, with 2 samples the compiler always
					// unrolls but it fails to keep the values in registers, working from the stack which
					// is inefficient.
					rtm::vector4f rotation_as_vec0 = rotations_as_vec[0];
					rtm::vector4f rotation_as_vec1 = rotations_as_vec[1];

					const uint32_t num_rotation_components = decomp_context.num_rotation_components;

					if (are_any_enum_flags_set(decomp_context.range_reduction, range_reduction_flags8::rotations))
					{
						if (decomp_context.has_segments)
						{
							const uint32_t segment_range_min_offset = sampling_context_.segment_range_data_offset;
							const uint32_t segment_range_extent_offset = sampling_context_.segment_range_data_offset + (num_rotation_components * sizeof(uint8_t));

							if (rotation_format == rotation_format8::quatf_drop_w_variable && decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_drop_w_variable))
							{
								if ((range_ignore_flags & 0x04) == 0)
								{
									const rtm::vector4f segment_range_min = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[0] + segment_range_min_offset);
									const rtm::vector4f segment_range_extent = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[0] + segment_range_extent_offset);

									rotation_as_vec0 = rtm::vector_mul_add(rotation_as_vec0, segment_range_extent, segment_range_min);
								}

								if ((range_ignore_flags & 0x01) == 0)
								{
									const rtm::vector4f segment_range_min = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[1] + segment_range_min_offset);
									const rtm::vector4f segment_range_extent = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[1] + segment_range_extent_offset);

									rotation_as_vec1 = rtm::vector_mul_add(rotation_as_vec1, segment_range_extent, segment_range_min);
								}
							}
							else
							{
								if (rotation_format == rotation_format8::quatf_full && decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_full))
								{
									{
										const rtm::vector4f segment_range_min = unpack_vector4_32(decomp_context.segment_range_data[0] + segment_range_min_offset, true);
										const rtm::vector4f segment_range_extent = unpack_vector4_32(decomp_context.segment_range_data[0] + segment_range_extent_offset, true);

										rotation_as_vec0 = rtm::vector_mul_add(rotation_as_vec0, segment_range_extent, segment_range_min);
									}

									{
										const rtm::vector4f segment_range_min = unpack_vector4_32(decomp_context.segment_range_data[1] + segment_range_min_offset, true);
										const rtm::vector4f segment_range_extent = unpack_vector4_32(decomp_context.segment_range_data[1] + segment_range_extent_offset, true);

										rotation_as_vec1 = rtm::vector_mul_add(rotation_as_vec1, segment_range_extent, segment_range_min);
									}
								}
								else
								{
									{
										const rtm::vector4f segment_range_min = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[0] + segment_range_min_offset);
										const rtm::vector4f segment_range_extent = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[0] + segment_range_extent_offset);

										rotation_as_vec0 = rtm::vector_mul_add(rotation_as_vec0, segment_range_extent, segment_range_min);
									}

									{
										const rtm::vector4f segment_range_min = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[1] + segment_range_min_offset);
										const rtm::vector4f segment_range_extent = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[1] + segment_range_extent_offset);

										rotation_as_vec1 = rtm::vector_mul_add(rotation_as_vec1, segment_range_extent, segment_range_min);
									}
								}
							}

							sampling_context_.segment_range_data_offset += num_rotation_components * k_segment_range_reduction_num_bytes_per_component * 2;
						}

						const rtm::vector4f clip_range_min = rtm::vector_load(decomp_context.clip_range_data + sampling_context_.clip_range_data_offset);
						const rtm::vector4f clip_range_extent = rtm::vector_load(decomp_context.clip_range_data + sampling_context_.clip_range_data_offset + (num_rotation_components * sizeof(float)));

						if ((range_ignore_flags & 0x08) == 0)
							rotation_as_vec0 = rtm::vector_mul_add(rotation_as_vec0, clip_range_extent, clip_range_min);

						if ((range_ignore_flags & 0x02) == 0)
							rotation_as_vec1 = rtm::vector_mul_add(rotation_as_vec1, clip_range_extent, clip_range_min);

						sampling_context_.clip_range_data_offset += num_rotation_components * sizeof(float) * 2;
					}

					// No-op conversion
					rtm::quatf rotation0 = rtm::vector_to_quat(rotation_as_vec0);
					rtm::quatf rotation1 = rtm::vector_to_quat(rotation_as_vec1);

					if (rotation_format != rotation_format8::quatf_full || !decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_full))
					{
						// We dropped the W component
						rotation0 = rtm::quat_from_positive_w(rotation_as_vec0);
						rotation1 = rtm::quat_from_positive_w(rotation_as_vec1);
					}

					const bool normalize_rotations = decompression_settings_type::normalize_rotations();
					if (normalize_rotations)
						interpolated_rotation = rtm::quat_lerp(rotation0, rotation1, decomp_context.interpolation_alpha);
					else
						interpolated_rotation = quat_lerp_no_normalization(rotation0, rotation1, decomp_context.interpolation_alpha);

					ACL_ASSERT(rtm::quat_is_finite(interpolated_rotation), "Rotation is not valid!");
					ACL_ASSERT(rtm::quat_is_normalized(interpolated_rotation) || !decompression_settings_type::normalize_rotations(), "Rotation is not normalized!");
				}
			}

			sampling_context_.track_index++;
			return interpolated_rotation;
		}

		template <class decompression_settings_type>
		inline rtm::quatf RTM_SIMD_CALL decompress_and_interpolate_animated_rotation(const persistent_transform_decompression_context_v0& decomp_context, sampling_context_v0& sampling_context_)
		{
			rtm::quatf interpolated_rotation;

			// This part is fairly complex, we'll loop and write to the stack (sampling context)
			rtm::vector4f* rotations_as_vec = &sampling_context_.vectors[0];

			// Range ignore flags are used to skip range normalization at the clip and/or segment levels
			// Each sample has two bits like so:
			//    - 0x01 = sample 1 segment
			//    - 0x02 = sample 1 clip
			//    - 0x04 = sample 0 segment
			//    - 0x08 = sample 0 clip
			// By default, we never ignore range reduction
			uint32_t range_ignore_flags = 0;

			const rotation_format8 rotation_format = get_rotation_format<decompression_settings_type>(decomp_context.rotation_format);
			if (rotation_format == rotation_format8::quatf_drop_w_variable && decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_drop_w_variable))
			{
				for (uint32_t i = 0; i < 2; ++i)
				{
					range_ignore_flags <<= 2;

					const uint8_t bit_rate = decomp_context.format_per_track_data[i][sampling_context_.format_per_track_data_offset];
					const uint32_t num_bits_at_bit_rate = get_num_bits_at_bit_rate(bit_rate);

					if (is_constant_bit_rate(bit_rate))
					{
						rotations_as_vec[i] = unpack_vector3_u48_unsafe(decomp_context.segment_range_data[i] + sampling_context_.segment_range_data_offset);
						range_ignore_flags |= 0x00000001U;	// Skip segment only
					}
					else if (is_raw_bit_rate(bit_rate))
					{
						rotations_as_vec[i] = unpack_vector3_96_unsafe(decomp_context.animated_track_data[i], sampling_context_.key_frame_bit_offsets[i]);
						range_ignore_flags |= 0x00000003U;	// Skip clip and segment
					}
					else
						rotations_as_vec[i] = unpack_vector3_uXX_unsafe(uint8_t(num_bits_at_bit_rate), decomp_context.animated_track_data[i], sampling_context_.key_frame_bit_offsets[i]);

					sampling_context_.key_frame_bit_offsets[i] += num_bits_at_bit_rate * 3;
				}

				sampling_context_.format_per_track_data_offset++;
			}
			else
			{
				if (rotation_format == rotation_format8::quatf_full && decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_full))
				{
					for (uint32_t i = 0; i < 2; ++i)
					{
						rotations_as_vec[i] = unpack_vector4_128_unsafe(decomp_context.animated_track_data[i], sampling_context_.key_frame_bit_offsets[i]);
						sampling_context_.key_frame_bit_offsets[i] += 128;
					}
				}
				else if (rotation_format == rotation_format8::quatf_drop_w_full && decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_drop_w_full))
				{
					for (uint32_t i = 0; i < 2; ++i)
					{
						rotations_as_vec[i] = unpack_vector3_96_unsafe(decomp_context.animated_track_data[i], sampling_context_.key_frame_bit_offsets[i]);
						sampling_context_.key_frame_bit_offsets[i] += 96;
					}
				}
			}

			// Load our samples to avoid working with the stack now that things can be unrolled.
			// We unroll because even if we work from the stack, with 2 samples the compiler always
			// unrolls but it fails to keep the values in registers, working from the stack which
			// is inefficient.
			rtm::vector4f rotation_as_vec0 = rotations_as_vec[0];
			rtm::vector4f rotation_as_vec1 = rotations_as_vec[1];

			const uint32_t num_rotation_components = decomp_context.num_rotation_components;

			if (are_any_enum_flags_set(decomp_context.range_reduction, range_reduction_flags8::rotations))
			{
				if (decomp_context.has_segments)
				{
					const uint32_t segment_range_min_offset = sampling_context_.segment_range_data_offset;
					const uint32_t segment_range_extent_offset = sampling_context_.segment_range_data_offset + (num_rotation_components * sizeof(uint8_t));

					if (rotation_format == rotation_format8::quatf_drop_w_variable && decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_drop_w_variable))
					{
						if ((range_ignore_flags & 0x04) == 0)
						{
							const rtm::vector4f segment_range_min = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[0] + segment_range_min_offset);
							const rtm::vector4f segment_range_extent = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[0] + segment_range_extent_offset);

							rotation_as_vec0 = rtm::vector_mul_add(rotation_as_vec0, segment_range_extent, segment_range_min);
						}

						if ((range_ignore_flags & 0x01) == 0)
						{
							const rtm::vector4f segment_range_min = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[1] + segment_range_min_offset);
							const rtm::vector4f segment_range_extent = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[1] + segment_range_extent_offset);

							rotation_as_vec1 = rtm::vector_mul_add(rotation_as_vec1, segment_range_extent, segment_range_min);
						}
					}
					else
					{
						if (rotation_format == rotation_format8::quatf_full && decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_full))
						{
							{
								const rtm::vector4f segment_range_min = unpack_vector4_32(decomp_context.segment_range_data[0] + segment_range_min_offset, true);
								const rtm::vector4f segment_range_extent = unpack_vector4_32(decomp_context.segment_range_data[0] + segment_range_extent_offset, true);

								rotation_as_vec0 = rtm::vector_mul_add(rotation_as_vec0, segment_range_extent, segment_range_min);
							}

							{
								const rtm::vector4f segment_range_min = unpack_vector4_32(decomp_context.segment_range_data[1] + segment_range_min_offset, true);
								const rtm::vector4f segment_range_extent = unpack_vector4_32(decomp_context.segment_range_data[1] + segment_range_extent_offset, true);

								rotation_as_vec1 = rtm::vector_mul_add(rotation_as_vec1, segment_range_extent, segment_range_min);
							}
						}
						else
						{
							{
								const rtm::vector4f segment_range_min = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[0] + segment_range_min_offset);
								const rtm::vector4f segment_range_extent = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[0] + segment_range_extent_offset);

								rotation_as_vec0 = rtm::vector_mul_add(rotation_as_vec0, segment_range_extent, segment_range_min);
							}

							{
								const rtm::vector4f segment_range_min = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[1] + segment_range_min_offset);
								const rtm::vector4f segment_range_extent = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[1] + segment_range_extent_offset);

								rotation_as_vec1 = rtm::vector_mul_add(rotation_as_vec1, segment_range_extent, segment_range_min);
							}
						}
					}

					sampling_context_.segment_range_data_offset += num_rotation_components * k_segment_range_reduction_num_bytes_per_component * 2;
				}

				const rtm::vector4f clip_range_min = rtm::vector_load(decomp_context.clip_range_data + sampling_context_.clip_range_data_offset);
				const rtm::vector4f clip_range_extent = rtm::vector_load(decomp_context.clip_range_data + sampling_context_.clip_range_data_offset + (num_rotation_components * sizeof(float)));

				if ((range_ignore_flags & 0x08) == 0)
					rotation_as_vec0 = rtm::vector_mul_add(rotation_as_vec0, clip_range_extent, clip_range_min);

				if ((range_ignore_flags & 0x02) == 0)
					rotation_as_vec1 = rtm::vector_mul_add(rotation_as_vec1, clip_range_extent, clip_range_min);

				sampling_context_.clip_range_data_offset += num_rotation_components * sizeof(float) * 2;
			}

			// No-op conversion
			rtm::quatf rotation0 = rtm::vector_to_quat(rotation_as_vec0);
			rtm::quatf rotation1 = rtm::vector_to_quat(rotation_as_vec1);

			if (rotation_format != rotation_format8::quatf_full || !decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_full))
			{
				// We dropped the W component
				rotation0 = rtm::quat_from_positive_w(rotation_as_vec0);
				rotation1 = rtm::quat_from_positive_w(rotation_as_vec1);
			}

			const bool normalize_rotations = decompression_settings_type::normalize_rotations();
			if (normalize_rotations)
				interpolated_rotation = rtm::quat_lerp(rotation0, rotation1, decomp_context.interpolation_alpha);
			else
				interpolated_rotation = quat_lerp_no_normalization(rotation0, rotation1, decomp_context.interpolation_alpha);

			ACL_ASSERT(rtm::quat_is_finite(interpolated_rotation), "Rotation is not valid!");
			ACL_ASSERT(rtm::quat_is_normalized(interpolated_rotation) || !decompression_settings_type::normalize_rotations(), "Rotation is not normalized!");

			return interpolated_rotation;
		}

		template<class decompression_settings_adapter_type>
		inline void skip_over_vector(const persistent_transform_decompression_context_v0& decomp_context, sampling_context_v0& sampling_context_)
		{
			const bitset_index_ref track_index_bit_ref(decomp_context.bitset_desc, sampling_context_.track_index);
			const bool is_sample_default = bitset_test(decomp_context.default_tracks_bitset, track_index_bit_ref);
			if (!is_sample_default)
			{
				const bool is_sample_constant = bitset_test(decomp_context.constant_tracks_bitset, track_index_bit_ref);
				if (is_sample_constant)
				{
					// Constant Vector3 tracks store the remaining sample with full precision
					sampling_context_.constant_track_data_offset += get_packed_vector_size(vector_format8::vector3f_full);
				}
				else
				{
					const vector_format8 format = get_vector_format<decompression_settings_adapter_type>(decompression_settings_adapter_type::get_vector_format(decomp_context));

					if (is_vector_format_variable(format))
					{
						for (uint32_t i = 0; i < 2; ++i)
						{
							const uint8_t bit_rate = decomp_context.format_per_track_data[i][sampling_context_.format_per_track_data_offset];
							const uint32_t num_bits_at_bit_rate = get_num_bits_at_bit_rate(bit_rate) * 3;	// 3 components

							sampling_context_.key_frame_bit_offsets[i] += num_bits_at_bit_rate;
						}

						sampling_context_.format_per_track_data_offset++;
					}
					else
					{
						for (uint32_t i = 0; i < 2; ++i)
							sampling_context_.key_frame_bit_offsets[i] += 96;
					}

					const range_reduction_flags8 range_reduction_flag = decompression_settings_adapter_type::get_range_reduction_flag();

					if (are_any_enum_flags_set(decomp_context.range_reduction, range_reduction_flag))
					{
						sampling_context_.clip_range_data_offset += k_clip_range_reduction_vector3_range_size;

						if (decomp_context.has_segments)
							sampling_context_.segment_range_data_offset += 3 * k_segment_range_reduction_num_bytes_per_component * 2;
					}
				}
			}

			sampling_context_.track_index++;
		}

		template<class decompression_settings_adapter_type>
		inline rtm::vector4f RTM_SIMD_CALL decompress_and_interpolate_vector(const persistent_transform_decompression_context_v0& decomp_context, rtm::vector4f_arg0 default_value, sampling_context_v0& sampling_context_)
		{
			rtm::vector4f interpolated_vector;

			const bitset_index_ref track_index_bit_ref(decomp_context.bitset_desc, sampling_context_.track_index);
			const bool is_sample_default = bitset_test(decomp_context.default_tracks_bitset, track_index_bit_ref);
			if (is_sample_default)
			{
				interpolated_vector = default_value;
			}
			else
			{
				const bool is_sample_constant = bitset_test(decomp_context.constant_tracks_bitset, track_index_bit_ref);
				if (is_sample_constant)
				{
					// Constant translation tracks store the remaining sample with full precision
					interpolated_vector = unpack_vector3_96_unsafe(decomp_context.constant_track_data + sampling_context_.constant_track_data_offset);
					ACL_ASSERT(rtm::vector_is_finite3(interpolated_vector), "Vector is not valid!");

					sampling_context_.constant_track_data_offset += get_packed_vector_size(vector_format8::vector3f_full);
				}
				else
				{
					const vector_format8 format = get_vector_format<decompression_settings_adapter_type>(decompression_settings_adapter_type::get_vector_format(decomp_context));

					// This part is fairly complex, we'll loop and write to the stack (sampling context)
					rtm::vector4f* vectors = &sampling_context_.vectors[0];

					// Range ignore flags are used to skip range normalization at the clip and/or segment levels
					// Each sample has two bits like so:
					//    - 0x01 = sample 1 segment
					//    - 0x02 = sample 1 clip
					//    - 0x04 = sample 0 segment
					//    - 0x08 = sample 0 clip
					// By default, we never ignore range reduction
					uint32_t range_ignore_flags = 0;

					if (format == vector_format8::vector3f_variable && decompression_settings_adapter_type::is_vector_format_supported(vector_format8::vector3f_variable))
					{
						for (uint32_t i = 0; i < 2; ++i)
						{
							range_ignore_flags <<= 2;

							const uint8_t bit_rate = decomp_context.format_per_track_data[i][sampling_context_.format_per_track_data_offset];
							const uint32_t num_bits_at_bit_rate = get_num_bits_at_bit_rate(bit_rate);

							if (is_constant_bit_rate(bit_rate))
							{
								vectors[i] = unpack_vector3_u48_unsafe(decomp_context.segment_range_data[i] + sampling_context_.segment_range_data_offset);
								range_ignore_flags |= 0x00000001U;	// Skip segment only
							}
							else if (is_raw_bit_rate(bit_rate))
							{
								vectors[i] = unpack_vector3_96_unsafe(decomp_context.animated_track_data[i], sampling_context_.key_frame_bit_offsets[i]);
								range_ignore_flags |= 0x00000003U;	// Skip clip and segment
							}
							else
								vectors[i] = unpack_vector3_uXX_unsafe(uint8_t(num_bits_at_bit_rate), decomp_context.animated_track_data[i], sampling_context_.key_frame_bit_offsets[i]);

							sampling_context_.key_frame_bit_offsets[i] += num_bits_at_bit_rate * 3;
						}

						sampling_context_.format_per_track_data_offset++;
					}
					else
					{
						if (format == vector_format8::vector3f_full && decompression_settings_adapter_type::is_vector_format_supported(vector_format8::vector3f_full))
						{
							for (uint32_t i = 0; i < 2; ++i)
							{
								vectors[i] = unpack_vector3_96_unsafe(decomp_context.animated_track_data[i], sampling_context_.key_frame_bit_offsets[i]);
								sampling_context_.key_frame_bit_offsets[i] += 96;
							}
						}
					}

					// Load our samples to avoid working with the stack now that things can be unrolled.
					// We unroll because even if we work from the stack, with 2 samples the compiler always
					// unrolls but it fails to keep the values in registers, working from the stack which
					// is inefficient.
					rtm::vector4f vector0 = vectors[0];
					rtm::vector4f vector1 = vectors[1];

					const range_reduction_flags8 range_reduction_flag = decompression_settings_adapter_type::get_range_reduction_flag();
					if (are_any_enum_flags_set(decomp_context.range_reduction, range_reduction_flag))
					{
						if (decomp_context.has_segments)
						{
							const uint32_t segment_range_min_offset = sampling_context_.segment_range_data_offset;
							const uint32_t segment_range_extent_offset = sampling_context_.segment_range_data_offset + (3 * sizeof(uint8_t));

							if ((range_ignore_flags & 0x04) == 0)
							{
								const rtm::vector4f segment_range_min = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[0] + segment_range_min_offset);
								const rtm::vector4f segment_range_extent = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[0] + segment_range_extent_offset);

								vector0 = rtm::vector_mul_add(vector0, segment_range_extent, segment_range_min);
							}

							if ((range_ignore_flags & 0x01) == 0)
							{
								const rtm::vector4f segment_range_min = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[1] + segment_range_min_offset);
								const rtm::vector4f segment_range_extent = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[1] + segment_range_extent_offset);

								vector1 = rtm::vector_mul_add(vector1, segment_range_extent, segment_range_min);
							}

							sampling_context_.segment_range_data_offset += 3 * k_segment_range_reduction_num_bytes_per_component * 2;
						}

						const rtm::vector4f clip_range_min = unpack_vector3_96_unsafe(decomp_context.clip_range_data + sampling_context_.clip_range_data_offset);
						const rtm::vector4f clip_range_extent = unpack_vector3_96_unsafe(decomp_context.clip_range_data + sampling_context_.clip_range_data_offset + (3 * sizeof(float)));

						if ((range_ignore_flags & 0x08) == 0)
							vector0 = rtm::vector_mul_add(vector0, clip_range_extent, clip_range_min);

						if ((range_ignore_flags & 0x02) == 0)
							vector1 = rtm::vector_mul_add(vector1, clip_range_extent, clip_range_min);

						sampling_context_.clip_range_data_offset += k_clip_range_reduction_vector3_range_size;
					}

					interpolated_vector = rtm::vector_lerp(vector0, vector1, decomp_context.interpolation_alpha);

					ACL_ASSERT(rtm::vector_is_finite3(interpolated_vector), "Vector is not valid!");
				}
			}

			sampling_context_.track_index++;
			return interpolated_vector;
		}

		template<class decompression_settings_adapter_type>
		inline rtm::vector4f RTM_SIMD_CALL decompress_and_interpolate_animated_vector3(const persistent_transform_decompression_context_v0& decomp_context, sampling_context_v0& sampling_context_)
		{
			rtm::vector4f interpolated_vector;

			const vector_format8 format = get_vector_format<decompression_settings_adapter_type>(decompression_settings_adapter_type::get_vector_format(decomp_context));

			// This part is fairly complex, we'll loop and write to the stack (sampling context)
			rtm::vector4f* vectors = &sampling_context_.vectors[0];

			// Range ignore flags are used to skip range normalization at the clip and/or segment levels
			// Each sample has two bits like so:
			//    - 0x01 = sample 1 segment
			//    - 0x02 = sample 1 clip
			//    - 0x04 = sample 0 segment
			//    - 0x08 = sample 0 clip
			// By default, we never ignore range reduction
			uint32_t range_ignore_flags = 0;

			if (format == vector_format8::vector3f_variable && decompression_settings_adapter_type::is_vector_format_supported(vector_format8::vector3f_variable))
			{
				for (uint32_t i = 0; i < 2; ++i)
				{
					range_ignore_flags <<= 2;

					const uint8_t bit_rate = decomp_context.format_per_track_data[i][sampling_context_.format_per_track_data_offset];
					const uint32_t num_bits_at_bit_rate = get_num_bits_at_bit_rate(bit_rate);

					if (is_constant_bit_rate(bit_rate))
					{
						vectors[i] = unpack_vector3_u48_unsafe(decomp_context.segment_range_data[i] + sampling_context_.segment_range_data_offset);
						range_ignore_flags |= 0x00000001U;	// Skip segment only
					}
					else if (is_raw_bit_rate(bit_rate))
					{
						vectors[i] = unpack_vector3_96_unsafe(decomp_context.animated_track_data[i], sampling_context_.key_frame_bit_offsets[i]);
						range_ignore_flags |= 0x00000003U;	// Skip clip and segment
					}
					else
						vectors[i] = unpack_vector3_uXX_unsafe(uint8_t(num_bits_at_bit_rate), decomp_context.animated_track_data[i], sampling_context_.key_frame_bit_offsets[i]);

					sampling_context_.key_frame_bit_offsets[i] += num_bits_at_bit_rate * 3;
				}

				sampling_context_.format_per_track_data_offset++;
			}
			else
			{
				if (format == vector_format8::vector3f_full && decompression_settings_adapter_type::is_vector_format_supported(vector_format8::vector3f_full))
				{
					for (uint32_t i = 0; i < 2; ++i)
					{
						vectors[i] = unpack_vector3_96_unsafe(decomp_context.animated_track_data[i], sampling_context_.key_frame_bit_offsets[i]);
						sampling_context_.key_frame_bit_offsets[i] += 96;
					}
				}
			}

			// Load our samples to avoid working with the stack now that things can be unrolled.
			// We unroll because even if we work from the stack, with 2 samples the compiler always
			// unrolls but it fails to keep the values in registers, working from the stack which
			// is inefficient.
			rtm::vector4f vector0 = vectors[0];
			rtm::vector4f vector1 = vectors[1];

			const range_reduction_flags8 range_reduction_flag = decompression_settings_adapter_type::get_range_reduction_flag();
			if (are_any_enum_flags_set(decomp_context.range_reduction, range_reduction_flag))
			{
				if (decomp_context.has_segments)
				{
					const uint32_t segment_range_min_offset = sampling_context_.segment_range_data_offset;
					const uint32_t segment_range_extent_offset = sampling_context_.segment_range_data_offset + (3 * sizeof(uint8_t));

					if ((range_ignore_flags & 0x04) == 0)
					{
						const rtm::vector4f segment_range_min = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[0] + segment_range_min_offset);
						const rtm::vector4f segment_range_extent = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[0] + segment_range_extent_offset);

						vector0 = rtm::vector_mul_add(vector0, segment_range_extent, segment_range_min);
					}

					if ((range_ignore_flags & 0x01) == 0)
					{
						const rtm::vector4f segment_range_min = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[1] + segment_range_min_offset);
						const rtm::vector4f segment_range_extent = unpack_vector3_u24_unsafe(decomp_context.segment_range_data[1] + segment_range_extent_offset);

						vector1 = rtm::vector_mul_add(vector1, segment_range_extent, segment_range_min);
					}

					sampling_context_.segment_range_data_offset += 3 * k_segment_range_reduction_num_bytes_per_component * 2;
				}

				const rtm::vector4f clip_range_min = unpack_vector3_96_unsafe(decomp_context.clip_range_data + sampling_context_.clip_range_data_offset);
				const rtm::vector4f clip_range_extent = unpack_vector3_96_unsafe(decomp_context.clip_range_data + sampling_context_.clip_range_data_offset + (3 * sizeof(float)));

				if ((range_ignore_flags & 0x08) == 0)
					vector0 = rtm::vector_mul_add(vector0, clip_range_extent, clip_range_min);

				if ((range_ignore_flags & 0x02) == 0)
					vector1 = rtm::vector_mul_add(vector1, clip_range_extent, clip_range_min);

				sampling_context_.clip_range_data_offset += k_clip_range_reduction_vector3_range_size;
			}

			interpolated_vector = rtm::vector_lerp(vector0, vector1, decomp_context.interpolation_alpha);

			ACL_ASSERT(rtm::vector_is_finite3(interpolated_vector), "Vector is not valid!");

			return interpolated_vector;
		}

		template<class decompression_settings_type>
		inline bool initialize_v0(persistent_transform_decompression_context_v0& context, const compressed_tracks& tracks)
		{
			ACL_ASSERT(tracks.get_algorithm_type() == algorithm_type8::uniformly_sampled, "Invalid algorithm type [%s], expected [%s]", get_algorithm_name(tracks.get_algorithm_type()), get_algorithm_name(algorithm_type8::uniformly_sampled));

			using translation_adapter = acl_impl::translation_decompression_settings_adapter<decompression_settings_type>;
			using scale_adapter = acl_impl::scale_decompression_settings_adapter<decompression_settings_type>;

			const tracks_header& header = get_tracks_header(tracks);
			const transform_tracks_header& transform_header = get_transform_tracks_header(tracks);

			const rotation_format8 packed_rotation_format = header.get_rotation_format();
			const vector_format8 packed_translation_format = header.get_translation_format();
			const vector_format8 packed_scale_format = header.get_scale_format();
			const rotation_format8 rotation_format = get_rotation_format<decompression_settings_type>(packed_rotation_format);
			const vector_format8 translation_format = get_vector_format<translation_adapter>(packed_translation_format);
			const vector_format8 scale_format = get_vector_format<scale_adapter>(packed_scale_format);

			ACL_ASSERT(rotation_format == packed_rotation_format, "Statically compiled rotation format (%s) differs from the compressed rotation format (%s)!", get_rotation_format_name(rotation_format), get_rotation_format_name(packed_rotation_format));
			ACL_ASSERT(translation_format == packed_translation_format, "Statically compiled translation format (%s) differs from the compressed translation format (%s)!", get_vector_format_name(translation_format), get_vector_format_name(packed_translation_format));
			ACL_ASSERT(scale_format == packed_scale_format, "Statically compiled scale format (%s) differs from the compressed scale format (%s)!", get_vector_format_name(scale_format), get_vector_format_name(packed_scale_format));

			context.tracks = &tracks;
			context.clip_hash = tracks.get_hash();
			context.clip_duration = calculate_duration(header.num_samples, header.sample_rate);
			context.sample_time = -1.0F;
			context.default_tracks_bitset = transform_header.get_default_tracks_bitset();

			context.constant_tracks_bitset = transform_header.get_constant_tracks_bitset();
			context.constant_track_data = transform_header.get_constant_track_data();
			context.clip_range_data = transform_header.get_clip_range_data();

			for (uint32_t key_frame_index = 0; key_frame_index < 2; ++key_frame_index)
			{
				context.format_per_track_data[key_frame_index] = nullptr;
				context.segment_range_data[key_frame_index] = nullptr;
				context.animated_track_data[key_frame_index] = nullptr;
			}

			const bool has_scale = header.get_has_scale();
			const uint32_t num_tracks_per_bone = has_scale ? 3 : 2;
			context.bitset_desc = bitset_description::make_from_num_bits(header.num_tracks * num_tracks_per_bone);

			range_reduction_flags8 range_reduction = range_reduction_flags8::none;
			if (is_rotation_format_variable(rotation_format))
				range_reduction |= range_reduction_flags8::rotations;
			if (is_vector_format_variable(translation_format))
				range_reduction |= range_reduction_flags8::translations;
			if (is_vector_format_variable(scale_format))
				range_reduction |= range_reduction_flags8::scales;

			context.rotation_format = rotation_format;
			context.translation_format = translation_format;
			context.scale_format = scale_format;
			context.range_reduction = range_reduction;
			context.num_rotation_components = rotation_format == rotation_format8::quatf_full ? 4 : 3;
			context.has_segments = transform_header.num_segments > 1;

			return true;
		}

		inline bool is_dirty_v0(const persistent_transform_decompression_context_v0& context, const compressed_tracks& tracks)
		{
			if (context.tracks != &tracks)
				return true;

			if (context.clip_hash != tracks.get_hash())
				return true;

			return false;
		}

		template<class decompression_settings_type>
		inline void seek_v0(persistent_transform_decompression_context_v0& context, float sample_time, sample_rounding_policy rounding_policy)
		{
			// Clamp for safety, the caller should normally handle this but in practice, it often isn't the case
			if (decompression_settings_type::clamp_sample_time())
				sample_time = rtm::scalar_clamp(sample_time, 0.0F, context.clip_duration);

			if (context.sample_time == sample_time)
				return;

			context.sample_time = sample_time;

			const tracks_header& header = get_tracks_header(*context.tracks);
			const transform_tracks_header& transform_header = get_transform_tracks_header(*context.tracks);

			uint32_t key_frame0;
			uint32_t key_frame1;
			find_linear_interpolation_samples_with_sample_rate(header.num_samples, header.sample_rate, sample_time, rounding_policy, key_frame0, key_frame1, context.interpolation_alpha);

			uint32_t segment_key_frame0;
			uint32_t segment_key_frame1;

			const segment_header* segment_header0;
			const segment_header* segment_header1;

			const segment_header* segment_headers = transform_header.get_segment_headers();
			const uint32_t num_segments = transform_header.num_segments;

			if (num_segments == 1)
			{
				// Key frame 0 and 1 are in the only segment present
				// This is a really common case and when it happens, we don't store the segment start index (zero)
				segment_header0 = segment_headers;
				segment_key_frame0 = key_frame0;

				segment_header1 = segment_headers;
				segment_key_frame1 = key_frame1;
			}
			else
			{
				const uint32_t* segment_start_indices = transform_header.get_segment_start_indices();

				// See segment_streams(..) for implementation details. This implementation is directly tied to it.
				const uint32_t approx_num_samples_per_segment = header.num_samples / num_segments;	// TODO: Store in header?
				const uint32_t approx_segment_index = key_frame0 / approx_num_samples_per_segment;

				uint32_t segment_index0 = 0;
				uint32_t segment_index1 = 0;

				// Our approximate segment guess is just that, a guess. The actual segments we need could be just before or after.
				// We start looking one segment earlier and up to 2 after. If we have too few segments after, we will hit the
				// sentinel value of 0xFFFFFFFF and exit the loop.
				// TODO: Can we do this with SIMD? Load all 4 values, set key_frame0, compare, move mask, count leading zeroes
				const uint32_t start_segment_index = approx_segment_index > 0 ? (approx_segment_index - 1) : 0;
				const uint32_t end_segment_index = start_segment_index + 4;

				for (uint32_t segment_index = start_segment_index; segment_index < end_segment_index; ++segment_index)
				{
					if (key_frame0 < segment_start_indices[segment_index])
					{
						// We went too far, use previous segment
						ACL_ASSERT(segment_index > 0, "Invalid segment index: %u", segment_index);
						segment_index0 = segment_index - 1;
						segment_index1 = key_frame1 < segment_start_indices[segment_index] ? segment_index0 : segment_index;
						break;
					}
				}

				segment_header0 = segment_headers + segment_index0;
				segment_header1 = segment_headers + segment_index1;

				segment_key_frame0 = key_frame0 - segment_start_indices[segment_index0];
				segment_key_frame1 = key_frame1 - segment_start_indices[segment_index1];
			}

			transform_header.get_segment_data(*segment_header0, context.format_per_track_data[0], context.segment_range_data[0], context.animated_track_data[0]);

			// More often than not the two segments are identical, when this is the case, just copy our pointers
			if (segment_header0 != segment_header1)
				transform_header.get_segment_data(*segment_header1, context.format_per_track_data[1], context.segment_range_data[1], context.animated_track_data[1]);
			else
			{
				context.format_per_track_data[1] = context.format_per_track_data[0];
				context.segment_range_data[1] = context.segment_range_data[0];
				context.animated_track_data[1] = context.animated_track_data[0];
			}

			context.key_frame_bit_offsets[0] = segment_key_frame0 * segment_header0->animated_pose_bit_size;
			context.key_frame_bit_offsets[1] = segment_key_frame1 * segment_header1->animated_pose_bit_size;
		}

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

		template<class decompression_settings_type>
		ACL_FORCE_INLINE void unpack_constant_quat(const persistent_transform_decompression_context_v0& decomp_context, track_cache_v0<rtm::quatf>& track_cache, const uint8_t*& constant_data)
		{
			uint32_t num_left_to_unpack = track_cache.num_left_to_unpack;
			if (num_left_to_unpack == 0)
				return;	// Nothing left to do, we are done

			// If we have less than 4 cached samples, unpack 4 more and prefetch the next cache line
			const uint32_t num_cached = track_cache.get_num_cached();
			if (num_cached >= 4)
				return;	// Enough cached, nothing to do

			const rotation_format8 rotation_format = get_rotation_format<decompression_settings_type>(decomp_context.rotation_format);

			const uint32_t num_to_unpack = std::min<uint32_t>(num_left_to_unpack, 4);
			num_left_to_unpack -= num_to_unpack;
			track_cache.num_left_to_unpack = num_left_to_unpack;

			// Write index will be either 0 or 4 here since we always unpack 4 at a time
			uint32_t cache_write_index = track_cache.cache_write_index % 8;
			track_cache.cache_write_index += num_to_unpack;

			const uint8_t* constant_track_data = constant_data;

			if (rotation_format == rotation_format8::quatf_full && decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_full))
			{
				for (uint32_t unpack_index = num_to_unpack; unpack_index != 0; --unpack_index)
				{
					// Unpack
					const rtm::quatf sample = unpack_quat_128(constant_track_data);

					ACL_ASSERT(rtm::quat_is_finite(sample), "Rotation is not valid!");
					ACL_ASSERT(rtm::quat_is_normalized(sample), "Rotation is not normalized!");

					// Cache
					track_cache.cached_samples[cache_write_index] = sample;
					cache_write_index++;

					// Update our read ptr
					constant_track_data += sizeof(rtm::float4f);
				}
			}
			else
			{
				// Unpack
				// Always load 4x rotations, we might contain garbage in a few lanes but it's fine
				const uint32_t load_size = num_to_unpack * sizeof(float);

				const rtm::vector4f xxxx = rtm::vector_load(reinterpret_cast<const float*>(constant_track_data + load_size * 0));
				const rtm::vector4f yyyy = rtm::vector_load(reinterpret_cast<const float*>(constant_track_data + load_size * 1));
				const rtm::vector4f zzzz = rtm::vector_load(reinterpret_cast<const float*>(constant_track_data + load_size * 2));

				// Update our read ptr
				constant_track_data += load_size * 3;

				// quat_from_positive_w_soa
				const rtm::vector4f wwww_squared = rtm::vector_sub(rtm::vector_sub(rtm::vector_sub(rtm::vector_set(1.0F), rtm::vector_mul(xxxx, xxxx)), rtm::vector_mul(yyyy, yyyy)), rtm::vector_mul(zzzz, zzzz));

				// w_squared can be negative either due to rounding or due to quantization imprecision, we take the absolute value
				// to ensure the resulting quaternion is always normalized with a positive W component
				const rtm::vector4f wwww = rtm::vector_sqrt(rtm::vector_abs(wwww_squared));

				// Swizzle out our 4 samples
				const rtm::vector4f tmp0 = rtm::vector_mix<rtm::mix4::x, rtm::mix4::y, rtm::mix4::a, rtm::mix4::b>(xxxx, yyyy);
				const rtm::vector4f tmp1 = rtm::vector_mix<rtm::mix4::z, rtm::mix4::w, rtm::mix4::c, rtm::mix4::d>(xxxx, yyyy);
				const rtm::vector4f tmp2 = rtm::vector_mix<rtm::mix4::x, rtm::mix4::y, rtm::mix4::a, rtm::mix4::b>(zzzz, wwww);
				const rtm::vector4f tmp3 = rtm::vector_mix<rtm::mix4::z, rtm::mix4::w, rtm::mix4::c, rtm::mix4::d>(zzzz, wwww);

				const rtm::vector4f sample_0 = rtm::vector_mix<rtm::mix4::x, rtm::mix4::z, rtm::mix4::a, rtm::mix4::c>(tmp0, tmp2);
				const rtm::vector4f sample_1 = rtm::vector_mix<rtm::mix4::y, rtm::mix4::w, rtm::mix4::b, rtm::mix4::d>(tmp0, tmp2);
				const rtm::vector4f sample_2 = rtm::vector_mix<rtm::mix4::x, rtm::mix4::z, rtm::mix4::a, rtm::mix4::c>(tmp1, tmp3);
				const rtm::vector4f sample_3 = rtm::vector_mix<rtm::mix4::y, rtm::mix4::w, rtm::mix4::b, rtm::mix4::d>(tmp1, tmp3);

				// Cache
				rtm::quatf* cache_ptr = &track_cache.cached_samples[cache_write_index];
				cache_ptr[0] = rtm::vector_to_quat(sample_0);
				cache_ptr[1] = rtm::vector_to_quat(sample_1);
				cache_ptr[2] = rtm::vector_to_quat(sample_2);
				cache_ptr[3] = rtm::vector_to_quat(sample_3);

#if defined(ACL_HAS_ASSERT_CHECKS)
				for (uint32_t unpack_index = 0; unpack_index < num_to_unpack; ++unpack_index)
				{
					ACL_ASSERT(rtm::quat_is_finite(track_cache.cached_samples[cache_write_index + unpack_index]), "Rotation is not valid!");
					ACL_ASSERT(rtm::quat_is_normalized(track_cache.cached_samples[cache_write_index + unpack_index]), "Rotation is not normalized!");
				}
#endif
			}

			// Update our pointer
			constant_data = constant_track_data;

			// Prefetch the next cache line even if we don't have any data left
			// By the time we unpack again, it will have arrived in the CPU cache
			// If our format is full precision, we have at most 4 samples per cache line
			// If our format is drop W, we have at most 5.33 samples per cache line

			// If our pointer was already aligned to a cache line before we unpacked our 4 values,
			// it now points to the first byte of the next cache line. Any offset between 0-63 will fetch it.
			// If our pointer had some offset into a cache line, we might have spanned 2 cache lines.
			// If this happens, we probably already read some data from the next cache line in which
			// case we don't need to prefetch it and we can go to the next one. Any offset after the end
			// of this cache line will fetch it. For safety, we prefetch 63 bytes ahead.
			// Prefetch 4 samples ahead in all levels of the CPU cache
			ACL_IMPL_CONSTANT_PREFETCH(constant_track_data + 63);
		}

#if defined(ACL_IMPL_VEC3_UNPACK)
		inline void unpack_constant_vector3(track_cache_v0<rtm::vector4f>& track_cache, const uint8_t*& constant_data)
		{
			uint32_t num_left_to_unpack = track_cache.num_left_to_unpack;
			if (num_left_to_unpack == 0)
				return;	// Nothing left to do, we are done

			const uint32_t packed_size = get_packed_vector_size(vector_format8::vector3f_full);

			// If we have less than 4 cached samples, unpack 4 more and prefetch the next cache line
			const uint32_t num_cached = track_cache.get_num_cached();
			if (num_cached < 4)
			{
				const uint32_t num_to_unpack = std::min<uint32_t>(num_left_to_unpack, 4);
				num_left_to_unpack -= num_to_unpack;
				track_cache.num_left_to_unpack = num_left_to_unpack;

				// Write index will be either 0 or 4 here since we always unpack 4 at a time
				uint32_t cache_write_index = track_cache.cache_write_index % 8;
				track_cache.cache_write_index += num_to_unpack;

				const uint8_t* constant_track_data = constant_data;

				for (uint32_t unpack_index = num_to_unpack; unpack_index != 0; --unpack_index)
				{
					// Unpack
					// Constant vector3 tracks store the remaining sample with full precision
					const rtm::vector4f sample = unpack_vector3_96_unsafe(constant_track_data);
					ACL_ASSERT(rtm::vector_is_finite3(sample), "Vector3 is not valid!");

					// TODO: Fill in W component with something sensible?

					// Cache
					track_cache.cached_samples[cache_write_index] = sample;
					cache_write_index++;

					// Update our read ptr
					constant_track_data += packed_size;
				}

				constant_data = constant_track_data;

				// Prefetch the next cache line even if we don't have any data left
				// By the time we unpack again, it will have arrived in the CPU cache
				// With our full precision format, we have at most 5.33 samples per cache line

				// If our pointer was already aligned to a cache line before we unpacked our 4 values,
				// it now points to the first byte of the next cache line. Any offset between 0-63 will fetch it.
				// If our pointer had some offset into a cache line, we might have spanned 2 cache lines.
				// If this happens, we probably already read some data from the next cache line in which
				// case we don't need to prefetch it and we can go to the next one. Any offset after the end
				// of this cache line will fetch it. For safety, we prefetch 63 bytes ahead.
				// Prefetch 4 samples ahead in all levels of the CPU cache
				ACL_IMPL_CONSTANT_PREFETCH(constant_track_data + 63);
			}
		}
#endif

		struct constant_track_cache_v0
		{
			track_cache_v0<rtm::quatf> rotations;

#if defined(ACL_IMPL_VEC3_UNPACK)
			track_cache_v0<rtm::vector4f> translations;
			track_cache_v0<rtm::vector4f> scales;
#endif

#if defined(ACL_IMPL_USE_CONSTANT_GROUPS)
			// How many we have left to unpack in total
			uint32_t		num_left_to_unpack_translations;
			uint32_t		num_left_to_unpack_scales;

			// How many we have cached (faked for translations/scales)
			uint32_t		num_unpacked_translations = 0;
			uint32_t		num_unpacked_scales = 0;

			// How many we have left in our group
			uint32_t		num_group_translations[2];
			uint32_t		num_group_scales[2];

			const uint8_t*	constant_data;
			const uint8_t*	constant_data_translations[2];
			const uint8_t*	constant_data_scales[2];
#else
			// Points to our packed sub-track data
			const uint8_t*	constant_data_rotations;
			const uint8_t*	constant_data_translations;
			const uint8_t*	constant_data_scales;
#endif

			template<class decompression_settings_type>
			void initialize(const persistent_transform_decompression_context_v0& decomp_context)
			{
				const transform_tracks_header& transform_header = get_transform_tracks_header(*decomp_context.tracks);

				rotations.num_left_to_unpack = transform_header.num_constant_rotation_samples;

#if defined(ACL_IMPL_VEC3_UNPACK)
				translations.num_left_to_unpack = transform_header.num_constant_translation_samples;
				scales.num_left_to_unpack = transform_header.num_constant_scale_samples;
#endif

#if defined(ACL_IMPL_USE_CONSTANT_GROUPS)
				num_left_to_unpack_translations = transform_header.num_constant_translation_samples;
				num_left_to_unpack_scales = transform_header.num_constant_scale_samples;

				constant_data = decomp_context.constant_track_data;
				constant_data_translations[0] = constant_data_translations[1] = nullptr;
				constant_data_scales[0] = constant_data_scales[1] = nullptr;
				num_group_translations[0] = num_group_translations[1] = 0;
				num_group_scales[0] = num_group_scales[1] = 0;
#else
				const rotation_format8 rotation_format = get_rotation_format<decompression_settings_type>(decomp_context.rotation_format);
				const rotation_format8 packed_format = is_rotation_format_variable(rotation_format) ? get_highest_variant_precision(get_rotation_variant(rotation_format)) : rotation_format;
				const uint32_t packed_rotation_size = get_packed_rotation_size(packed_format);
				const uint32_t packed_translation_size = get_packed_vector_size(vector_format8::vector3f_full);

				constant_data_rotations = decomp_context.constant_track_data;
				constant_data_translations = constant_data_rotations + packed_rotation_size * transform_header.num_constant_rotation_samples;
				constant_data_scales = constant_data_translations + packed_translation_size * transform_header.num_constant_translation_samples;
#endif
			}

			template<class decompression_settings_type>
			ACL_FORCE_INLINE void unpack_rotations(const persistent_transform_decompression_context_v0& decomp_context)
			{
#if defined(ACL_IMPL_USE_CONSTANT_GROUPS)
				unpack_constant_quat<decompression_settings_type>(decomp_context, rotations, constant_data);
#else
				unpack_constant_quat<decompression_settings_type>(decomp_context, rotations, constant_data_rotations);
#endif
			}

			rtm::quatf RTM_SIMD_CALL consume_rotation()
			{
				ACL_ASSERT(rotations.cache_read_index < rotations.cache_write_index, "Attempting to consume a constant sample that isn't cached");
				const uint32_t cache_read_index = rotations.cache_read_index++;
				return rotations.cached_samples[cache_read_index % 8];
			}

			void unpack_translations()
			{
#if defined(ACL_IMPL_VEC3_UNPACK)
				unpack_constant_vector3(translations, constant_data_translations);
#else
#if defined(ACL_IMPL_USE_CONSTANT_GROUPS)
				if (num_left_to_unpack_translations == 0 || num_unpacked_translations >= 4)
					return;	// Enough unpacked or nothing to do

				const uint32_t num_to_unpack = std::min<uint32_t>(num_left_to_unpack_translations, 4);
				num_left_to_unpack_translations -= num_to_unpack;

				// If we have data already unpacked, store in index 1 otherwise store in 0
				const uint32_t unpack_index = num_unpacked_translations > 0 ? 1 : 0;
				constant_data_translations[unpack_index] = constant_data;
				num_group_translations[unpack_index] = num_to_unpack;
				constant_data += sizeof(rtm::float3f) * num_to_unpack;

				num_unpacked_translations += num_to_unpack;

				ACL_IMPL_CONSTANT_PREFETCH(constant_data + 63);
#else
				ACL_IMPL_CONSTANT_PREFETCH(constant_data_translations + 63);
#endif
#endif
			}

			rtm::vector4f RTM_SIMD_CALL consume_translation()
			{
#if defined(ACL_IMPL_VEC3_UNPACK)
				ACL_ASSERT(translations.cache_read_index < translations.cache_write_index, "Attempting to consume a constant sample that isn't cached");
				const uint32_t cache_read_index = translations.cache_read_index++;
				return translations.cached_samples[cache_read_index % 8];
#else
				

#if defined(ACL_IMPL_USE_CONSTANT_GROUPS)
				const rtm::vector4f translation = rtm::vector_load(constant_data_translations[0]);
				num_group_translations[0]--;
				num_unpacked_translations--;

				// If we finished reading from the first group, swap it out otherwise increment our entry
				if (num_group_translations[0] == 0)
				{
					constant_data_translations[0] = constant_data_translations[1];
					num_group_translations[0] = num_group_translations[1];
				}
				else
					constant_data_translations[0] += sizeof(rtm::float3f);
#else
				const rtm::vector4f translation = rtm::vector_load(constant_data_translations);
				constant_data_translations += sizeof(rtm::float3f);
#endif
				return translation;
#endif
			}

			void unpack_scales()
			{
#if defined(ACL_IMPL_VEC3_UNPACK)
				unpack_constant_vector3(scales, constant_data_scales);
#else
#if defined(ACL_IMPL_USE_CONSTANT_GROUPS)
				if (num_left_to_unpack_scales == 0 || num_unpacked_scales >= 4)
					return;	// Enough unpacked or nothing to do

				const uint32_t num_to_unpack = std::min<uint32_t>(num_left_to_unpack_scales, 4);
				num_left_to_unpack_scales -= num_to_unpack;

				// If we have data already unpacked, store in index 1 otherwise store in 0
				const uint32_t unpack_index = num_unpacked_scales > 0 ? 1 : 0;
				constant_data_scales[unpack_index] = constant_data;
				num_group_scales[unpack_index] = num_to_unpack;
				constant_data += sizeof(rtm::float3f) * num_to_unpack;

				num_unpacked_scales += num_to_unpack;

				ACL_IMPL_CONSTANT_PREFETCH(constant_data + 63);
#else
				ACL_IMPL_CONSTANT_PREFETCH(constant_data_scales + 63);
#endif
#endif
			}

			rtm::vector4f RTM_SIMD_CALL consume_scale()
			{
#if defined(ACL_IMPL_VEC3_UNPACK)
				ACL_ASSERT(scales.cache_read_index < scales.cache_write_index, "Attempting to consume a constant sample that isn't cached");
				const uint32_t cache_read_index = scales.cache_read_index++;
				return scales.cached_samples[cache_read_index % 8];
#else
#if defined(ACL_IMPL_USE_CONSTANT_GROUPS)
				const rtm::vector4f scale = rtm::vector_load(constant_data_scales[0]);
				num_group_scales[0]--;
				num_unpacked_scales--;

				// If we finished reading from the first group, swap it out otherwise increment our entry
				if (num_group_scales[0] == 0)
				{
					constant_data_scales[0] = constant_data_scales[1];
					num_group_scales[0] = num_group_scales[1];
				}
				else
					constant_data_scales[0] += sizeof(rtm::float3f);
#else
				const rtm::vector4f scale = rtm::vector_load(constant_data_scales);
				constant_data_scales += sizeof(rtm::float3f);
#endif
				return scale;
#endif
			}
		};

		struct clip_animated_sampling_context_v0
		{
			// Data is ordered in groups of 4 animated sub-tracks (e.g rot0, rot1, rot2, rot3)
			// Order depends on animated track order. If we have 6 animated rotation tracks before the first animated
			// translation track, we'll have 8 animated rotation sub-tracks followed by 4 animated translation sub-tracks.
			// Once we reach the end, there is no extra padding. The last group might be less than 4 sub-tracks.
			// This is because we always process 4 animated sub-tracks at a time and cache the results.

			const uint8_t* clip_range_data;				// Range information of the current sub-track in the clip
		};

		struct segment_animated_sampling_context_v0
		{
			// Data is ordered in groups of 4 animated sub-tracks (e.g rot0, rot1, rot2, rot3)
			// Order depends on animated track order. If we have 6 animated rotation tracks before the first animated
			// translation track, we'll have 8 animated rotation sub-tracks followed by 4 animated translation sub-tracks.
			// Once we reach the end, there is no extra padding. The last group might be less than 4 sub-tracks.
			// This is because we always process 4 animated sub-tracks at a time and cache the results.

			const uint8_t* format_per_track_data;		// Metadata of the current sub-track
			const uint8_t* segment_range_data;			// Range information (or constant sample if bit rate is 0) of the current sub-track in this segment

			// For the animated samples, constant bit rate sub-tracks (with a bit rate of 0) do not contain samples.
			// As such, their group will not contain 4 sub-tracks.

			const uint8_t* animated_track_data;			// Base of animated sample data, constant and doesn't change after init
			uint32_t animated_track_data_bit_offset;	// Bit offset of the current animated sub-track
		};

		template<class decompression_settings_type>
		inline ACL_DISABLE_SECURITY_COOKIE_CHECK void unpack_animated_quat(const persistent_transform_decompression_context_v0& decomp_context, rtm::vector4f output_scratch[4],
			uint32_t num_to_unpack,
			clip_animated_sampling_context_v0& clip_sampling_context, segment_animated_sampling_context_v0& segment_sampling_context)
		{
			const rotation_format8 rotation_format = get_rotation_format<decompression_settings_type>(decomp_context.rotation_format);

			uint32_t segment_range_ignore_mask = 0;
			uint32_t clip_range_ignore_mask = 0;

			const uint8_t* format_per_track_data = segment_sampling_context.format_per_track_data;
			const uint8_t* segment_range_data = segment_sampling_context.segment_range_data;
			const uint8_t* animated_track_data = segment_sampling_context.animated_track_data;
			uint32_t animated_track_data_bit_offset = segment_sampling_context.animated_track_data_bit_offset;

			for (uint32_t unpack_index = 0; unpack_index < num_to_unpack; ++unpack_index)
			{
				// Our decompressed rotation as a vector4
				rtm::vector4f rotation_as_vec;

				if (rotation_format == rotation_format8::quatf_drop_w_variable && decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_drop_w_variable))
				{
					const uint32_t bit_rate = *format_per_track_data;
					format_per_track_data++;

					uint32_t sample_segment_range_ignore_mask;
					uint32_t sample_clip_range_ignore_mask;
					if (is_constant_bit_rate(bit_rate))
					{
						rotation_as_vec = unpack_vector3_u48_unsafe(segment_range_data);
						sample_segment_range_ignore_mask = 0xFF;
						sample_clip_range_ignore_mask = 0x00;
					}
					else if (is_raw_bit_rate(bit_rate))
					{
						rotation_as_vec = unpack_vector3_96_unsafe(animated_track_data, animated_track_data_bit_offset);
						animated_track_data_bit_offset += 96;
						sample_segment_range_ignore_mask = 0xFF;
						sample_clip_range_ignore_mask = 0xFF;
					}
					else
					{
						const uint32_t num_bits_at_bit_rate = get_num_bits_at_bit_rate(bit_rate);
						rotation_as_vec = unpack_vector3_uXX_unsafe(num_bits_at_bit_rate, animated_track_data, animated_track_data_bit_offset);
						animated_track_data_bit_offset += num_bits_at_bit_rate * 3;
						sample_segment_range_ignore_mask = 0x00;
						sample_clip_range_ignore_mask = 0x00;
					}

					// Skip constant sample stored in segment range data
					// Raw bit rates have unused range data, skip it
					// Skip segment range data
					// If we have no segments, we have no range data and do not have constant bit rates
					segment_range_data += sizeof(uint16_t) * 3;

					// Masks are used in little endian format so the first sample is in the LSB end
					segment_range_ignore_mask |= sample_segment_range_ignore_mask << (unpack_index * 8);
					clip_range_ignore_mask |= sample_clip_range_ignore_mask << (unpack_index * 8);
				}
				else
				{
					if (rotation_format == rotation_format8::quatf_full && decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_full))
					{
						rotation_as_vec = unpack_vector4_128_unsafe(animated_track_data, animated_track_data_bit_offset);
						animated_track_data_bit_offset += 128;
					}
					else // rotation_format8::quatf_drop_w_full
					{
						rotation_as_vec = unpack_vector3_96_unsafe(animated_track_data, animated_track_data_bit_offset);
						animated_track_data_bit_offset += 96;
					}
				}

				output_scratch[unpack_index] = rotation_as_vec;
			}

			// Prefetch the next cache line even if we don't have any data left
			// By the time we unpack again, it will have arrived in the CPU cache
			// If our format is full precision, we have at most 4 samples per cache line
			// If our format is drop W, we have at most 5.33 samples per cache line

			// If our pointer was already aligned to a cache line before we unpacked our 4 values,
			// it now points to the first byte of the next cache line. Any offset between 0-63 will fetch it.
			// If our pointer had some offset into a cache line, we might have spanned 2 cache lines.
			// If this happens, we probably already read some data from the next cache line in which
			// case we don't need to prefetch it and we can go to the next one. Any offset after the end
			// of this cache line will fetch it. For safety, we prefetch 63 bytes ahead.
			// Prefetch 4 samples ahead in all levels of the CPU cache
			ACL_IMPL_ANIMATED_PREFETCH(format_per_track_data + 63);
			ACL_IMPL_ANIMATED_PREFETCH(animated_track_data + (animated_track_data_bit_offset / 8) + 63);

			// Update our ptr
			segment_sampling_context.format_per_track_data = format_per_track_data;
			segment_sampling_context.animated_track_data_bit_offset = animated_track_data_bit_offset;

			// Swizzle our samples into SOA form
			rtm::vector4f tmp0 = rtm::vector_mix<rtm::mix4::x, rtm::mix4::y, rtm::mix4::a, rtm::mix4::b>(output_scratch[0], output_scratch[1]);
			rtm::vector4f tmp1 = rtm::vector_mix<rtm::mix4::z, rtm::mix4::w, rtm::mix4::c, rtm::mix4::d>(output_scratch[0], output_scratch[1]);
			rtm::vector4f tmp2 = rtm::vector_mix<rtm::mix4::x, rtm::mix4::y, rtm::mix4::a, rtm::mix4::b>(output_scratch[2], output_scratch[3]);
			rtm::vector4f tmp3 = rtm::vector_mix<rtm::mix4::z, rtm::mix4::w, rtm::mix4::c, rtm::mix4::d>(output_scratch[2], output_scratch[3]);

			rtm::vector4f sample_xxxx = rtm::vector_mix<rtm::mix4::x, rtm::mix4::z, rtm::mix4::a, rtm::mix4::c>(tmp0, tmp2);
			rtm::vector4f sample_yyyy = rtm::vector_mix<rtm::mix4::y, rtm::mix4::w, rtm::mix4::b, rtm::mix4::d>(tmp0, tmp2);
			rtm::vector4f sample_zzzz = rtm::vector_mix<rtm::mix4::x, rtm::mix4::z, rtm::mix4::a, rtm::mix4::c>(tmp1, tmp3);

			// Reset our segment range data
			segment_range_data = segment_sampling_context.segment_range_data;

			if (rotation_format == rotation_format8::quatf_drop_w_variable && decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_drop_w_variable))
			{
				// TODO: Move range remapping out of here and do it with AVX together with quat W reconstruction

				const rtm::vector4f zero_v = rtm::vector_zero();
				const rtm::vector4f one_v = rtm::vector_set(1.0F);

#if defined(RTM_SSE2_INTRINSICS)
				__m128i ignore_masks_v8 = _mm_set_epi32(0, 0, clip_range_ignore_mask, segment_range_ignore_mask);
				__m128i ignore_masks_v16 = _mm_unpacklo_epi8(ignore_masks_v8, ignore_masks_v8);
#elif defined(RTM_NEON_INTRINSICS)
				const int8x8_t ignore_masks_v8 = vcreate_s8((uint64_t(clip_range_ignore_mask) << 32) | segment_range_ignore_mask);
				const int16x8_t ignore_masks_v16 = vmovl_s8(ignore_masks_v8);
#else
#error todo
#endif

				// TODO: Swizzle to simplify unpacking
				if (decomp_context.has_segments)
				{
#if defined(RTM_SSE2_INTRINSICS)
					__m128i zero = _mm_setzero_si128();
					__m128i x8y8z8_x8y8z8_x8y8z8_x8y8z8_0 = _mm_loadu_si128((const __m128i*)segment_range_data);		// contains: min0, extent0, min1 extent1
					__m128i x8y8z8_x8y8z8_x8y8z8_x8y8z8_1 = _mm_loadu_si128((const __m128i*)(segment_range_data + 12));	// contains: min2, extent2, min3 extent3

					// TODO: prefetch here segment data

#if defined(RTM_SSE4_INTRINSICS)
					// TODO: Can we zero out our min at the same time?
					__m128i swizzle_mask = _mm_set_epi32(0, 0x0B050802, 0x0A040701, 0x09030600);

					// [min0.x, min1.x, extent0.x, extent1.x], [min0.y, min1.y, extent0.y, extent0.y], [min0.z, min1.z, extent0.z, extent1.z]
					__m128i x8x8x8x8_y8y8y8y8_z8z8z8z8_0 = _mm_shuffle_epi8(x8y8z8_x8y8z8_x8y8z8_x8y8z8_0, swizzle_mask);
					// [min2.x, min3.x, extent2.x, extent3.x], [min2.y, min3.y, extent2.y, extent3.y], [min2.z, min3.z, extent2.z, extent3.z]
					__m128i x8x8x8x8_y8y8y8y8_z8z8z8z8_1 = _mm_shuffle_epi8(x8y8z8_x8y8z8_x8y8z8_x8y8z8_1, swizzle_mask);

					// [min0.x, min1.x, extent0.x, extent1.x], [min0.y, min1.y, extent0.y, extent0.y]
					__m128i x16x16x16x16_y16y16y16y16_0 = _mm_unpacklo_epi8(x8x8x8x8_y8y8y8y8_z8z8z8z8_0, zero);
					// [min2.x, min3.x, extent2.x, extent3.x], [min2.y, min3.y, extent2.y, extent3.y]
					__m128i x16x16x16x16_y16y16y16y16_1 = _mm_unpacklo_epi8(x8x8x8x8_y8y8y8y8_z8z8z8z8_1, zero);

					// [min0.z, min1.z, extent0.z, extent1.z]
					__m128i z16z16z16z16_0 = _mm_unpackhi_epi8(x8x8x8x8_y8y8y8y8_z8z8z8z8_0, zero);
					// [min2.z, min3.z, extent2.z, extent3.z]
					__m128i z16z16z16z16_1 = _mm_unpackhi_epi8(x8x8x8x8_y8y8y8y8_z8z8z8z8_1, zero);

					// [min0.x, min1.x, extent0.x, extent1.x]
					__m128i x32x32x32x32_0 = _mm_unpacklo_epi16(x16x16x16x16_y16y16y16y16_0, zero);
					// [min2.x, min3.x, extent2.x, extent3.x]
					__m128i x32x32x32x32_1 = _mm_unpacklo_epi16(x16x16x16x16_y16y16y16y16_1, zero);

					// [min0.y, min1.y, extent0.y, extent0.y]
					__m128i y32y32y32y32_0 = _mm_unpackhi_epi16(x16x16x16x16_y16y16y16y16_0, zero);
					// [min2.y, min3.y, extent2.y, extent3.y]
					__m128i y32y32y32y32_1 = _mm_unpackhi_epi16(x16x16x16x16_y16y16y16y16_1, zero);

					// [min0.z, min1.z, extent0.z, extent1.z]
					__m128i z32z32z32z32_0 = _mm_unpacklo_epi16(z16z16z16z16_0, zero);
					// [min2.z, min3.z, extent2.z, extent3.z]
					__m128i z32z32z32z32_1 = _mm_unpacklo_epi16(z16z16z16z16_1, zero);

					__m128 min0_x_min1_x_extent0_x_extent1_x = _mm_cvtepi32_ps(x32x32x32x32_0);
					__m128 min2_x_min3_x_extent2_x_extent3_x = _mm_cvtepi32_ps(x32x32x32x32_1);

					__m128 min0_y_min1_y_extent0_y_extent1_y = _mm_cvtepi32_ps(y32y32y32y32_0);
					__m128 min2_y_min3_y_extent2_y_extent3_y = _mm_cvtepi32_ps(y32y32y32y32_1);

					__m128 min0_z_min1_z_extent0_z_extent1_z = _mm_cvtepi32_ps(z32z32z32z32_0);
					__m128 min2_z_min3_z_extent2_z_extent3_z = _mm_cvtepi32_ps(z32z32z32z32_1);

					__m128 segment_range_min_xxxx = _mm_shuffle_ps(min0_x_min1_x_extent0_x_extent1_x, min2_x_min3_x_extent2_x_extent3_x, _MM_SHUFFLE(1, 0, 1, 0));
					__m128 segment_range_min_yyyy = _mm_shuffle_ps(min0_y_min1_y_extent0_y_extent1_y, min2_y_min3_y_extent2_y_extent3_y, _MM_SHUFFLE(1, 0, 1, 0));
					__m128 segment_range_min_zzzz = _mm_shuffle_ps(min0_z_min1_z_extent0_z_extent1_z, min2_z_min3_z_extent2_z_extent3_z, _MM_SHUFFLE(1, 0, 1, 0));

					__m128 segment_range_extent_xxxx = _mm_shuffle_ps(min0_x_min1_x_extent0_x_extent1_x, min2_x_min3_x_extent2_x_extent3_x, _MM_SHUFFLE(3, 2, 3, 2));
					__m128 segment_range_extent_yyyy = _mm_shuffle_ps(min0_y_min1_y_extent0_y_extent1_y, min2_y_min3_y_extent2_y_extent3_y, _MM_SHUFFLE(3, 2, 3, 2));
					__m128 segment_range_extent_zzzz = _mm_shuffle_ps(min0_z_min1_z_extent0_z_extent1_z, min2_z_min3_z_extent2_z_extent3_z, _MM_SHUFFLE(3, 2, 3, 2));

					__m128 normalization_value = _mm_set_ps1(1.0F / 255.0F);

					segment_range_min_xxxx = _mm_mul_ps(segment_range_min_xxxx, normalization_value);
					segment_range_min_yyyy = _mm_mul_ps(segment_range_min_yyyy, normalization_value);
					segment_range_min_zzzz = _mm_mul_ps(segment_range_min_zzzz, normalization_value);

					segment_range_extent_xxxx = _mm_mul_ps(segment_range_extent_xxxx, normalization_value);
					segment_range_extent_yyyy = _mm_mul_ps(segment_range_extent_yyyy, normalization_value);
					segment_range_extent_zzzz = _mm_mul_ps(segment_range_extent_zzzz, normalization_value);
#else
					__m128i x16y16z16_x16y16z16_x16y16_0 = _mm_unpacklo_epi8(x8y8z8_x8y8z8_x8y8z8_x8y8z8_0, zero);		// contains: min0, extent0, min1.xy
					__m128i z16_x16y16z16_0 = _mm_unpackhi_epi8(x8y8z8_x8y8z8_x8y8z8_x8y8z8_0, zero);					// contains: min1.z, extent1

					__m128i x16y16z16_x16y16z16_x16y16_1 = _mm_unpacklo_epi8(x8y8z8_x8y8z8_x8y8z8_x8y8z8_1, zero);		// contains: min2, extent2, min3.xy
					__m128i z16_x16y16z16_1 = _mm_unpackhi_epi8(x8y8z8_x8y8z8_x8y8z8_x8y8z8_1, zero);					// contains: min3.z, extent3

					__m128i x32y32z32_x32_0 = _mm_unpacklo_epi16(x16y16z16_x16y16z16_x16y16_0, zero);					// contains: min0, extent0.x
					__m128i y32z32_x32y32_0 = _mm_unpackhi_epi16(x16y16z16_x16y16z16_x16y16_0, zero);					// contains: extent0.yz, min1.xy
					__m128i z32_x32y32z32_0 = _mm_unpacklo_epi16(z16_x16y16z16_0, zero);								// contains: min1.z, extent1

					__m128i x32y32z32_x32_1 = _mm_unpacklo_epi16(x16y16z16_x16y16z16_x16y16_1, zero);					// contains: min2, extent2.x
					__m128i y32z32_x32y32_1 = _mm_unpackhi_epi16(x16y16z16_x16y16z16_x16y16_1, zero);					// contains: extent2.yz, min3.xy
					__m128i z32_x32y32z32_1 = _mm_unpacklo_epi16(z16_x16y16z16_1, zero);								// contains: min3.z, extent3

					__m128 x32y32z32_x32f_0 = _mm_cvtepi32_ps(x32y32z32_x32_0);											// contains: min0.xyz, extent0.x
					__m128 y32z32_x32y32f_0 = _mm_cvtepi32_ps(y32z32_x32y32_0);											// contains: extent0.yz, min1.xy
					__m128 z32_x32y32z32f_0 = _mm_cvtepi32_ps(z32_x32y32z32_0);											// contains: min1.z, extent1.xyz

					__m128 x32y32z32_x32f_1 = _mm_cvtepi32_ps(x32y32z32_x32_1);											// contains: min2.xyz, extent2.x
					__m128 y32z32_x32y32f_1 = _mm_cvtepi32_ps(y32z32_x32y32_1);											// contains: extent2.yz, min3.xy
					__m128 z32_x32y32z32f_1 = _mm_cvtepi32_ps(z32_x32y32z32_1);											// contains: min3.z, extent3.xyz

					__m128 normalization_value = _mm_set_ps1(1.0F / 255.0F);

					x32y32z32_x32f_0 = _mm_mul_ps(x32y32z32_x32f_0, normalization_value);
					y32z32_x32y32f_0 = _mm_mul_ps(y32z32_x32y32f_0, normalization_value);
					z32_x32y32z32f_0 = _mm_mul_ps(z32_x32y32z32f_0, normalization_value);

					x32y32z32_x32f_1 = _mm_mul_ps(x32y32z32_x32f_1, normalization_value);
					y32z32_x32y32f_1 = _mm_mul_ps(y32z32_x32y32f_1, normalization_value);
					z32_x32y32z32f_1 = _mm_mul_ps(z32_x32y32z32f_1, normalization_value);

					// Swizzle our segment range into SOA form
					__m128 min0_x_extent0_x_min1_x_extent0_y = _mm_shuffle_ps(x32y32z32_x32f_0, y32z32_x32y32f_0, _MM_SHUFFLE(0, 2, 3, 0));	// min0.x, extent0.x, min1.y, extent0.y
					__m128 min2_x_extent2_x_min3_x_extent2_y = _mm_shuffle_ps(x32y32z32_x32f_1, y32z32_x32y32f_1, _MM_SHUFFLE(0, 2, 3, 0)); // min2.x, extent2.x, min3.x, extent2.y

					__m128 min0_z_extent0_x_min1_z_extent1_x = _mm_shuffle_ps(x32y32z32_x32f_0, z32_x32y32z32f_0, _MM_SHUFFLE(1, 0, 3, 2)); // min0.z, extent0.x, min1.z, extent1.x
					__m128 min2_z_extent2_x_min3_z_extent3_x = _mm_shuffle_ps(x32y32z32_x32f_1, z32_x32y32z32f_1, _MM_SHUFFLE(1, 0, 3, 2)); // min2.z, extent2.x, min3.z, extent3.x

					__m128 min0_y_min0_z_min1_yy = _mm_shuffle_ps(x32y32z32_x32f_0, y32z32_x32y32f_0, _MM_SHUFFLE(3, 3, 2, 1));
					__m128 min2_y_min2_z_min3_yy = _mm_shuffle_ps(x32y32z32_x32f_1, y32z32_x32y32f_1, _MM_SHUFFLE(3, 3, 2, 1));

					__m128 extent0_y_extent0_z_extent1_y_extent1_z = _mm_shuffle_ps(y32z32_x32y32f_0, z32_x32y32z32f_0, _MM_SHUFFLE(3, 2, 1, 0));
					__m128 extent2_y_extent2_z_extent3_y_extent3_z = _mm_shuffle_ps(y32z32_x32y32f_1, z32_x32y32z32f_1, _MM_SHUFFLE(3, 2, 1, 0));

					__m128 segment_range_min_xxxx = _mm_shuffle_ps(min0_x_extent0_x_min1_x_extent0_y, min2_x_extent2_x_min3_x_extent2_y, _MM_SHUFFLE(2, 0, 2, 0));
					__m128 segment_range_min_yyyy = _mm_shuffle_ps(min0_y_min0_z_min1_yy, min2_y_min2_z_min3_yy, _MM_SHUFFLE(2, 0, 2, 0));
					__m128 segment_range_min_zzzz = _mm_shuffle_ps(min0_z_extent0_x_min1_z_extent1_x, min2_z_extent2_x_min3_z_extent3_x, _MM_SHUFFLE(2, 0, 2, 0));

					__m128 segment_range_extent_xxxx = _mm_shuffle_ps(min0_z_extent0_x_min1_z_extent1_x, min2_z_extent2_x_min3_z_extent3_x, _MM_SHUFFLE(3, 1, 3, 1));
					__m128 segment_range_extent_yyyy = _mm_shuffle_ps(extent0_y_extent0_z_extent1_y_extent1_z, extent2_y_extent2_z_extent3_y_extent3_z, _MM_SHUFFLE(2, 0, 2, 0));
					__m128 segment_range_extent_zzzz = _mm_shuffle_ps(extent0_y_extent0_z_extent1_y_extent1_z, extent2_y_extent2_z_extent3_y_extent3_z, _MM_SHUFFLE(3, 1, 3, 1));
#endif

					__m128i segment_range_ignore_mask_v32 = _mm_unpacklo_epi16(ignore_masks_v16, ignore_masks_v16);
					__m128 segment_range_ignore_mask_v32f = _mm_castsi128_ps(segment_range_ignore_mask_v32);

					// Mask out the segment ranges we ignore
					segment_range_min_xxxx = _mm_andnot_ps(segment_range_ignore_mask_v32f, segment_range_min_xxxx);
					segment_range_min_yyyy = _mm_andnot_ps(segment_range_ignore_mask_v32f, segment_range_min_yyyy);
					segment_range_min_zzzz = _mm_andnot_ps(segment_range_ignore_mask_v32f, segment_range_min_zzzz);
#elif defined(RTM_NEON_INTRINSICS)
					// [min0.xyz, extent0.xyz, min1.xyz, extent1.xyz, min2.xyz, extent2.xyz, min3.xyz, extent3.xyz] = 24 bytes
					const uint8x16_t segment_range_bytes_0_16 = vld1q_u8(segment_range_data);
					const uint8x8_t segment_range_bytes_16_24 = vld1_u8(segment_range_data + 16);

					uint8x8x3_t segment_range_bytes;
					segment_range_bytes.val[0] = vget_low_u8(segment_range_bytes_0_16);
					segment_range_bytes.val[1] = vget_high_u8(segment_range_bytes_0_16);
					segment_range_bytes.val[2] = segment_range_bytes_16_24;

					// TODO: Load first mask and offset, add offset for next masks
					const uint8x8_t swizzle_mask_min_x0x1 = vcreate_u8((0xFFFFFF06ULL << 32) | 0xFFFFFF00ULL);
					const uint8x8_t swizzle_mask_min_x2x3 = vcreate_u8((0xFFFFFF12ULL << 32) | 0xFFFFFF0CULL);
					const uint8x8_t swizzle_mask_min_y0y1 = vcreate_u8((0xFFFFFF07ULL << 32) | 0xFFFFFF01ULL);
					const uint8x8_t swizzle_mask_min_y2y3 = vcreate_u8((0xFFFFFF13ULL << 32) | 0xFFFFFF0DULL);
					const uint8x8_t swizzle_mask_min_z0z1 = vcreate_u8((0xFFFFFF08ULL << 32) | 0xFFFFFF02ULL);
					const uint8x8_t swizzle_mask_min_z2z3 = vcreate_u8((0xFFFFFF14ULL << 32) | 0xFFFFFF0EULL);

					const uint8x8_t swizzle_mask_extent_x0x1 = vcreate_u8((0xFFFFFF09ULL << 32) | 0xFFFFFF03ULL);
					const uint8x8_t swizzle_mask_extent_x2x3 = vcreate_u8((0xFFFFFF15ULL << 32) | 0xFFFFFF0FULL);
					const uint8x8_t swizzle_mask_extent_y0y1 = vcreate_u8((0xFFFFFF0AULL << 32) | 0xFFFFFF04ULL);
					const uint8x8_t swizzle_mask_extent_y2y3 = vcreate_u8((0xFFFFFF16ULL << 32) | 0xFFFFFF10ULL);
					const uint8x8_t swizzle_mask_extent_z0z1 = vcreate_u8((0xFFFFFF0BULL << 32) | 0xFFFFFF05ULL);
					const uint8x8_t swizzle_mask_extent_z2z3 = vcreate_u8((0xFFFFFF17ULL << 32) | 0xFFFFFF11ULL);

					const uint8x8_t segment_range_min_x0x1 = vtbl3_u8(segment_range_bytes, swizzle_mask_min_x0x1);
					const uint8x8_t segment_range_min_x2x3 = vtbl3_u8(segment_range_bytes, swizzle_mask_min_x2x3);
					const uint8x8_t segment_range_min_y0y1 = vtbl3_u8(segment_range_bytes, swizzle_mask_min_y0y1);
					const uint8x8_t segment_range_min_y2y3 = vtbl3_u8(segment_range_bytes, swizzle_mask_min_y2y3);
					const uint8x8_t segment_range_min_z0z1 = vtbl3_u8(segment_range_bytes, swizzle_mask_min_z0z1);
					const uint8x8_t segment_range_min_z2z3 = vtbl3_u8(segment_range_bytes, swizzle_mask_min_z2z3);

					const uint8x8_t segment_range_extent_x0x1 = vtbl3_u8(segment_range_bytes, swizzle_mask_extent_x0x1);
					const uint8x8_t segment_range_extent_x2x3 = vtbl3_u8(segment_range_bytes, swizzle_mask_extent_x2x3);
					const uint8x8_t segment_range_extent_y0y1 = vtbl3_u8(segment_range_bytes, swizzle_mask_extent_y0y1);
					const uint8x8_t segment_range_extent_y2y3 = vtbl3_u8(segment_range_bytes, swizzle_mask_extent_y2y3);
					const uint8x8_t segment_range_extent_z0z1 = vtbl3_u8(segment_range_bytes, swizzle_mask_extent_z0z1);
					const uint8x8_t segment_range_extent_z2z3 = vtbl3_u8(segment_range_bytes, swizzle_mask_extent_z2z3);

					uint32x4_t segment_range_min_xxxx_u32 = vreinterpretq_u32_u8(vcombine_u8(segment_range_min_x0x1, segment_range_min_x2x3));
					uint32x4_t segment_range_min_yyyy_u32 = vreinterpretq_u32_u8(vcombine_u8(segment_range_min_y0y1, segment_range_min_y2y3));
					uint32x4_t segment_range_min_zzzz_u32 = vreinterpretq_u32_u8(vcombine_u8(segment_range_min_z0z1, segment_range_min_z2z3));

					uint32x4_t segment_range_extent_xxxx_u32 = vreinterpretq_u32_u8(vcombine_u8(segment_range_extent_x0x1, segment_range_extent_x2x3));
					uint32x4_t segment_range_extent_yyyy_u32 = vreinterpretq_u32_u8(vcombine_u8(segment_range_extent_y0y1, segment_range_extent_y2y3));
					uint32x4_t segment_range_extent_zzzz_u32 = vreinterpretq_u32_u8(vcombine_u8(segment_range_extent_z0z1, segment_range_extent_z2z3));

					const uint32x4_t segment_range_ignore_mask_u32 = vreinterpretq_u32_s32(vmovl_s16(vget_low_s16(ignore_masks_v16)));
					const float32x4_t segment_range_ignore_mask_v32f = vreinterpretq_f32_u32(segment_range_ignore_mask_u32);

					// Mask out the segment ranges we ignore
					segment_range_min_xxxx_u32 = vbicq_u32(segment_range_min_xxxx_u32, segment_range_ignore_mask_u32);
					segment_range_min_yyyy_u32 = vbicq_u32(segment_range_min_yyyy_u32, segment_range_ignore_mask_u32);
					segment_range_min_zzzz_u32 = vbicq_u32(segment_range_min_zzzz_u32, segment_range_ignore_mask_u32);

					float32x4_t segment_range_min_xxxx = vcvtq_f32_u32(segment_range_min_xxxx_u32);
					float32x4_t segment_range_min_yyyy = vcvtq_f32_u32(segment_range_min_yyyy_u32);
					float32x4_t segment_range_min_zzzz = vcvtq_f32_u32(segment_range_min_zzzz_u32);

					float32x4_t segment_range_extent_xxxx = vcvtq_f32_u32(segment_range_extent_xxxx_u32);
					float32x4_t segment_range_extent_yyyy = vcvtq_f32_u32(segment_range_extent_yyyy_u32);
					float32x4_t segment_range_extent_zzzz = vcvtq_f32_u32(segment_range_extent_zzzz_u32);

					const float normalization_value = 1.0F / 255.0F;

					segment_range_min_xxxx = vmulq_n_f32(segment_range_min_xxxx, normalization_value);
					segment_range_min_yyyy = vmulq_n_f32(segment_range_min_yyyy, normalization_value);
					segment_range_min_zzzz = vmulq_n_f32(segment_range_min_zzzz, normalization_value);

					segment_range_extent_xxxx = vmulq_n_f32(segment_range_extent_xxxx, normalization_value);
					segment_range_extent_yyyy = vmulq_n_f32(segment_range_extent_yyyy, normalization_value);
					segment_range_extent_zzzz = vmulq_n_f32(segment_range_extent_zzzz, normalization_value);
#else
#error todo
#endif

					segment_range_extent_xxxx = rtm::vector_select(segment_range_ignore_mask_v32f, one_v, segment_range_extent_xxxx);
					segment_range_extent_yyyy = rtm::vector_select(segment_range_ignore_mask_v32f, one_v, segment_range_extent_yyyy);
					segment_range_extent_zzzz = rtm::vector_select(segment_range_ignore_mask_v32f, one_v, segment_range_extent_zzzz);

					sample_xxxx = rtm::vector_mul_add(sample_xxxx, segment_range_extent_xxxx, segment_range_min_xxxx);
					sample_yyyy = rtm::vector_mul_add(sample_yyyy, segment_range_extent_yyyy, segment_range_min_yyyy);
					sample_zzzz = rtm::vector_mul_add(sample_zzzz, segment_range_extent_zzzz, segment_range_min_zzzz);
				}

				const uint8_t* clip_range_data = clip_sampling_context.clip_range_data;

#if defined(RTM_SSE2_INTRINSICS)
				__m128i clip_range_ignore_mask_v32 = _mm_unpackhi_epi16(ignore_masks_v16, ignore_masks_v16);
				__m128 clip_range_ignore_mask_v32f = _mm_castsi128_ps(clip_range_ignore_mask_v32);
#elif defined(RTM_NEON_INTRINSICS)
				const uint32x4_t clip_range_ignore_mask_u32 = vreinterpretq_u32_s32(vmovl_s16(vget_high_s16(ignore_masks_v16)));
				const float32x4_t clip_range_ignore_mask_v32f = vreinterpretq_f32_u32(clip_range_ignore_mask_u32);
#else
#error todo
#endif

				// TODO: Swizzle the clip range data
				const rtm::vector4f clip_range_min0 = rtm::vector_load(clip_range_data + sizeof(rtm::float3f) * 0);
				const rtm::vector4f clip_range_min1 = rtm::vector_load(clip_range_data + sizeof(rtm::float3f) * 2);
				const rtm::vector4f clip_range_min2 = rtm::vector_load(clip_range_data + sizeof(rtm::float3f) * 4);
				const rtm::vector4f clip_range_min3 = rtm::vector_load(clip_range_data + sizeof(rtm::float3f) * 6);

				const rtm::vector4f clip_range_extent0 = rtm::vector_load(clip_range_data + sizeof(rtm::float3f) * 1);
				const rtm::vector4f clip_range_extent1 = rtm::vector_load(clip_range_data + sizeof(rtm::float3f) * 3);
				const rtm::vector4f clip_range_extent2 = rtm::vector_load(clip_range_data + sizeof(rtm::float3f) * 5);
				const rtm::vector4f clip_range_extent3 = rtm::vector_load(clip_range_data + sizeof(rtm::float3f) * 7);

				// Swizzle our samples into SOA form
				tmp0 = rtm::vector_mix<rtm::mix4::x, rtm::mix4::y, rtm::mix4::a, rtm::mix4::b>(clip_range_min0, clip_range_min1);
				tmp1 = rtm::vector_mix<rtm::mix4::z, rtm::mix4::w, rtm::mix4::c, rtm::mix4::d>(clip_range_min0, clip_range_min1);
				tmp2 = rtm::vector_mix<rtm::mix4::x, rtm::mix4::y, rtm::mix4::a, rtm::mix4::b>(clip_range_min2, clip_range_min3);
				tmp3 = rtm::vector_mix<rtm::mix4::z, rtm::mix4::w, rtm::mix4::c, rtm::mix4::d>(clip_range_min2, clip_range_min3);

				rtm::vector4f clip_range_min_xxxx = rtm::vector_mix<rtm::mix4::x, rtm::mix4::z, rtm::mix4::a, rtm::mix4::c>(tmp0, tmp2);
				rtm::vector4f clip_range_min_yyyy = rtm::vector_mix<rtm::mix4::y, rtm::mix4::w, rtm::mix4::b, rtm::mix4::d>(tmp0, tmp2);
				rtm::vector4f clip_range_min_zzzz = rtm::vector_mix<rtm::mix4::x, rtm::mix4::z, rtm::mix4::a, rtm::mix4::c>(tmp1, tmp3);

				tmp0 = rtm::vector_mix<rtm::mix4::x, rtm::mix4::y, rtm::mix4::a, rtm::mix4::b>(clip_range_extent0, clip_range_extent1);
				tmp1 = rtm::vector_mix<rtm::mix4::z, rtm::mix4::w, rtm::mix4::c, rtm::mix4::d>(clip_range_extent0, clip_range_extent1);
				tmp2 = rtm::vector_mix<rtm::mix4::x, rtm::mix4::y, rtm::mix4::a, rtm::mix4::b>(clip_range_extent2, clip_range_extent3);
				tmp3 = rtm::vector_mix<rtm::mix4::z, rtm::mix4::w, rtm::mix4::c, rtm::mix4::d>(clip_range_extent2, clip_range_extent3);

				rtm::vector4f clip_range_extent_xxxx = rtm::vector_mix<rtm::mix4::x, rtm::mix4::z, rtm::mix4::a, rtm::mix4::c>(tmp0, tmp2);
				rtm::vector4f clip_range_extent_yyyy = rtm::vector_mix<rtm::mix4::y, rtm::mix4::w, rtm::mix4::b, rtm::mix4::d>(tmp0, tmp2);
				rtm::vector4f clip_range_extent_zzzz = rtm::vector_mix<rtm::mix4::x, rtm::mix4::z, rtm::mix4::a, rtm::mix4::c>(tmp1, tmp3);

				// Mask out the clip ranges we ignore
#if defined(RTM_SSE2_INTRINSICS)
				clip_range_min_xxxx = _mm_andnot_ps(clip_range_ignore_mask_v32f, clip_range_min_xxxx);
				clip_range_min_yyyy = _mm_andnot_ps(clip_range_ignore_mask_v32f, clip_range_min_yyyy);
				clip_range_min_zzzz = _mm_andnot_ps(clip_range_ignore_mask_v32f, clip_range_min_zzzz);
#elif defined(RTM_NEON_INTRINSICS)
				clip_range_min_xxxx = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(clip_range_min_xxxx), clip_range_ignore_mask_u32));
				clip_range_min_yyyy = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(clip_range_min_yyyy), clip_range_ignore_mask_u32));
				clip_range_min_zzzz = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(clip_range_min_zzzz), clip_range_ignore_mask_u32));
#else
#error todo
#endif

				clip_range_extent_xxxx = rtm::vector_select(clip_range_ignore_mask_v32f, one_v, clip_range_extent_xxxx);
				clip_range_extent_yyyy = rtm::vector_select(clip_range_ignore_mask_v32f, one_v, clip_range_extent_yyyy);
				clip_range_extent_zzzz = rtm::vector_select(clip_range_ignore_mask_v32f, one_v, clip_range_extent_zzzz);

				sample_xxxx = rtm::vector_mul_add(sample_xxxx, clip_range_extent_xxxx, clip_range_min_xxxx);
				sample_yyyy = rtm::vector_mul_add(sample_yyyy, clip_range_extent_yyyy, clip_range_min_yyyy);
				sample_zzzz = rtm::vector_mul_add(sample_zzzz, clip_range_extent_zzzz, clip_range_min_zzzz);

				// Skip our used segment range data
				segment_range_data += sizeof(uint8_t) * 6 * num_to_unpack;

				// Update our ptr
				segment_sampling_context.segment_range_data = segment_range_data;

				// Prefetch the next cache line even if we don't have any data left
				// By the time we unpack again, it will have arrived in the CPU cache
				// If our format is full precision, we have at most 4 samples per cache line
				// If our format is drop W, we have at most 5.33 samples per cache line

				// If our pointer was already aligned to a cache line before we unpacked our 4 values,
				// it now points to the first byte of the next cache line. Any offset between 0-63 will fetch it.
				// If our pointer had some offset into a cache line, we might have spanned 2 cache lines.
				// If this happens, we probably already read some data from the next cache line in which
				// case we don't need to prefetch it and we can go to the next one. Any offset after the end
				// of this cache line will fetch it. For safety, we prefetch 63 bytes ahead.
				// Prefetch 4 samples ahead in all levels of the CPU cache
				ACL_IMPL_ANIMATED_PREFETCH(segment_range_data + 63);
			}
			else
			{
				rtm::vector4f sample_wwww = rtm::vector_mix<rtm::mix4::y, rtm::mix4::w, rtm::mix4::b, rtm::mix4::d>(tmp1, tmp3);
				output_scratch[3] = sample_wwww;
			}

			output_scratch[0] = sample_xxxx;
			output_scratch[1] = sample_yyyy;
			output_scratch[2] = sample_zzzz;
		}

		template<class decompression_settings_adapter_type>
		inline void unpack_animated_vector3(const persistent_transform_decompression_context_v0& decomp_context, rtm::vector4f output_scratch[4],
			uint32_t num_to_unpack,
			clip_animated_sampling_context_v0& clip_sampling_context, segment_animated_sampling_context_v0& segment_sampling_context)
		{
			const vector_format8 format = get_vector_format<decompression_settings_adapter_type>(decompression_settings_adapter_type::get_vector_format(decomp_context));

			for (uint32_t unpack_index = 0; unpack_index < num_to_unpack; ++unpack_index)
			{
				// Range ignore flags are used to skip range normalization at the clip and/or segment levels
				// Each sample has two bits like so:
				//    - 0x01 = ignore segment level
				//    - 0x02 = ignore clip level
				uint32_t range_ignore_flags;

				rtm::vector4f sample;
				if (format == vector_format8::vector3f_variable && decompression_settings_adapter_type::is_vector_format_supported(vector_format8::vector3f_variable))
				{
					const uint8_t bit_rate = *segment_sampling_context.format_per_track_data;
					segment_sampling_context.format_per_track_data++;

					if (is_constant_bit_rate(bit_rate))
					{
						sample = unpack_vector3_u48_unsafe(segment_sampling_context.segment_range_data);
						segment_sampling_context.segment_range_data += sizeof(uint16_t) * 3;
						range_ignore_flags = 0x01;	// Skip segment only
					}
					else if (is_raw_bit_rate(bit_rate))
					{
						sample = unpack_vector3_96_unsafe(segment_sampling_context.animated_track_data, segment_sampling_context.animated_track_data_bit_offset);
						segment_sampling_context.animated_track_data_bit_offset += 96;
						segment_sampling_context.segment_range_data += sizeof(uint16_t) * 3;	// Raw bit rates have unused range data, skip it
						range_ignore_flags = 0x03;	// Skip clip and segment
					}
					else
					{
						const uint32_t num_bits_at_bit_rate = get_num_bits_at_bit_rate(bit_rate);
						sample = unpack_vector3_uXX_unsafe(uint8_t(num_bits_at_bit_rate), segment_sampling_context.animated_track_data, segment_sampling_context.animated_track_data_bit_offset);
						segment_sampling_context.animated_track_data_bit_offset += num_bits_at_bit_rate * 3;
						range_ignore_flags = 0x00;	// Don't skip range reduction
					}
				}
				else // vector_format8::vector3f_full
				{
					sample = unpack_vector3_96_unsafe(segment_sampling_context.animated_track_data, segment_sampling_context.animated_track_data_bit_offset);
					segment_sampling_context.animated_track_data_bit_offset += 96;
					range_ignore_flags = 0x03;	// Skip clip and segment
				}

				if (decomp_context.has_segments && (range_ignore_flags & 0x01) == 0)
				{
					// Apply segment range remapping
					const uint32_t range_entry_size = 3 * sizeof(uint8_t);
					const uint8_t* segment_range_min_ptr = segment_sampling_context.segment_range_data;
					const uint8_t* segment_range_extent_ptr = segment_range_min_ptr + range_entry_size;
					segment_sampling_context.segment_range_data = segment_range_extent_ptr + range_entry_size;

					const rtm::vector4f segment_range_min = unpack_vector3_u24_unsafe(segment_range_min_ptr);
					const rtm::vector4f segment_range_extent = unpack_vector3_u24_unsafe(segment_range_extent_ptr);

					sample = rtm::vector_mul_add(sample, segment_range_extent, segment_range_min);
				}

				if ((range_ignore_flags & 0x02) == 0)
				{
					// Apply clip range remapping
					const uint32_t range_entry_size = 3 * sizeof(float);
					const uint32_t sub_track_offset = range_entry_size * 2 * unpack_index;
					const uint8_t* clip_range_min_ptr = clip_sampling_context.clip_range_data + sub_track_offset;
					const uint8_t* clip_range_extent_ptr = clip_range_min_ptr + range_entry_size;

					const rtm::vector4f clip_range_min = rtm::vector_load(clip_range_min_ptr);
					const rtm::vector4f clip_range_extent = rtm::vector_load(clip_range_extent_ptr);

					sample = rtm::vector_mul_add(sample, clip_range_extent, clip_range_min);
				}

				ACL_ASSERT(rtm::vector_is_finite3(sample), "Vector3 is not valid!");

				// TODO: Fill in W component with something sensible?

				// Cache
				output_scratch[unpack_index] = sample;
			}

			// Prefetch the next cache line even if we don't have any data left
			// By the time we unpack again, it will have arrived in the CPU cache
			// If our format is full precision, we have at most 4 samples per cache line
			// If our format is drop W, we have at most 5.33 samples per cache line

			// If our pointer was already aligned to a cache line before we unpacked our 4 values,
			// it now points to the first byte of the next cache line. Any offset between 0-63 will fetch it.
			// If our pointer had some offset into a cache line, we might have spanned 2 cache lines.
			// If this happens, we probably already read some data from the next cache line in which
			// case we don't need to prefetch it and we can go to the next one. Any offset after the end
			// of this cache line will fetch it. For safety, we prefetch 63 bytes ahead.
			// Prefetch 4 samples ahead in all levels of the CPU cache
			ACL_IMPL_ANIMATED_PREFETCH(segment_sampling_context.format_per_track_data + 63);
			ACL_IMPL_ANIMATED_PREFETCH(segment_sampling_context.animated_track_data + (segment_sampling_context.animated_track_data_bit_offset / 8) + 63);
			ACL_IMPL_ANIMATED_PREFETCH(segment_sampling_context.segment_range_data + 63);
		}

		struct animated_track_cache_v0
		{
			track_cache_v0<rtm::quatf> rotations;
			track_cache_v0<rtm::vector4f> translations;
			track_cache_v0<rtm::vector4f> scales;

			// Scratch space when we decompress our samples before we interpolate
			rtm::vector4f scratch0[4];
			rtm::vector4f scratch1[4];

			clip_animated_sampling_context_v0 clip_sampling_context;

			segment_animated_sampling_context_v0 segment_sampling_context[2];

			void initialize(const persistent_transform_decompression_context_v0& decomp_context)
			{
				clip_sampling_context.clip_range_data = decomp_context.clip_range_data;

				segment_sampling_context[0].format_per_track_data = decomp_context.format_per_track_data[0];
				segment_sampling_context[0].segment_range_data = decomp_context.segment_range_data[0];
				segment_sampling_context[0].animated_track_data = decomp_context.animated_track_data[0];
				segment_sampling_context[0].animated_track_data_bit_offset = decomp_context.key_frame_bit_offsets[0];

				segment_sampling_context[1].format_per_track_data = decomp_context.format_per_track_data[1];
				segment_sampling_context[1].segment_range_data = decomp_context.segment_range_data[1];
				segment_sampling_context[1].animated_track_data = decomp_context.animated_track_data[1];
				segment_sampling_context[1].animated_track_data_bit_offset = decomp_context.key_frame_bit_offsets[1];

				const transform_tracks_header& transform_header = get_transform_tracks_header(*decomp_context.tracks);

				rotations.num_left_to_unpack = transform_header.num_animated_rotation_sub_tracks;
				translations.num_left_to_unpack = transform_header.num_animated_translation_sub_tracks;
				scales.num_left_to_unpack = transform_header.num_animated_scale_sub_tracks;
			}

			template<class decompression_settings_type>
			void unpack_rotations(const persistent_transform_decompression_context_v0& decomp_context)
			{
				uint32_t num_left_to_unpack = rotations.num_left_to_unpack;
				if (num_left_to_unpack == 0)
					return;	// Nothing left to do, we are done

				// If we have less than 4 cached samples, unpack 4 more and prefetch the next cache line
				const uint32_t num_cached = rotations.get_num_cached();
				if (num_cached >= 4)
					return;	// Enough cached, nothing to do

				const uint32_t num_to_unpack = std::min<uint32_t>(num_left_to_unpack, 4);
				num_left_to_unpack -= num_to_unpack;
				rotations.num_left_to_unpack = num_left_to_unpack;

				// Write index will be either 0 or 4 here since we always unpack 4 at a time
				uint32_t cache_write_index = rotations.cache_write_index % 8;
				rotations.cache_write_index += num_to_unpack;

				unpack_animated_quat<decompression_settings_type>(decomp_context, scratch0, num_to_unpack, clip_sampling_context, segment_sampling_context[0]);
				unpack_animated_quat<decompression_settings_type>(decomp_context, scratch1, num_to_unpack, clip_sampling_context, segment_sampling_context[1]);

				// If we have a variable bit rate, we perform range reduction, skip the data we used
				const rotation_format8 rotation_format = get_rotation_format<decompression_settings_type>(decomp_context.rotation_format);
				if (rotation_format == rotation_format8::quatf_drop_w_variable && decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_drop_w_variable))
					clip_sampling_context.clip_range_data += num_to_unpack * sizeof(rtm::float3f) * 2;

				// Clip range data is 24-32 bytes per sub-track and as such we need to prefetch two cache lines ahead to process 4 sub-tracks
				ACL_IMPL_ANIMATED_PREFETCH(clip_sampling_context.clip_range_data + 63);
				ACL_IMPL_ANIMATED_PREFETCH(clip_sampling_context.clip_range_data + 127);

				rtm::vector4f scratch0_xxxx;
				rtm::vector4f scratch0_yyyy;
				rtm::vector4f scratch0_zzzz;
				rtm::vector4f scratch0_wwww;

				rtm::vector4f scratch1_xxxx;
				rtm::vector4f scratch1_yyyy;
				rtm::vector4f scratch1_zzzz;
				rtm::vector4f scratch1_wwww;

				// Reconstruct our quaternion W component in SOA
				if (rotation_format != rotation_format8::quatf_full || !decompression_settings_type::is_rotation_format_supported(rotation_format8::quatf_full))
				{
					// TODO: Use AVX for this
					scratch0_xxxx = scratch0[0];
					scratch0_yyyy = scratch0[1];
					scratch0_zzzz = scratch0[2];

					scratch1_xxxx = scratch1[0];
					scratch1_yyyy = scratch1[1];
					scratch1_zzzz = scratch1[2];

					// quat_from_positive_w_soa
					const rtm::vector4f scratch0_xxxx_squared = rtm::vector_mul(scratch0_xxxx, scratch0_xxxx);
					const rtm::vector4f scratch0_yyyy_squared = rtm::vector_mul(scratch0_yyyy, scratch0_yyyy);
					const rtm::vector4f scratch0_zzzz_squared = rtm::vector_mul(scratch0_zzzz, scratch0_zzzz);
					const rtm::vector4f scratch0_wwww_squared = rtm::vector_sub(rtm::vector_sub(rtm::vector_sub(rtm::vector_set(1.0F), scratch0_xxxx_squared), scratch0_yyyy_squared), scratch0_zzzz_squared);

					const rtm::vector4f scratch1_xxxx_squared = rtm::vector_mul(scratch1_xxxx, scratch1_xxxx);
					const rtm::vector4f scratch1_yyyy_squared = rtm::vector_mul(scratch1_yyyy, scratch1_yyyy);
					const rtm::vector4f scratch1_zzzz_squared = rtm::vector_mul(scratch1_zzzz, scratch1_zzzz);
					const rtm::vector4f scratch1_wwww_squared = rtm::vector_sub(rtm::vector_sub(rtm::vector_sub(rtm::vector_set(1.0F), scratch1_xxxx_squared), scratch1_yyyy_squared), scratch1_zzzz_squared);

					// w_squared can be negative either due to rounding or due to quantization imprecision, we take the absolute value
					// to ensure the resulting quaternion is always normalized with a positive W component
					scratch0_wwww = rtm::vector_sqrt(rtm::vector_abs(scratch0_wwww_squared));
					scratch1_wwww = rtm::vector_sqrt(rtm::vector_abs(scratch1_wwww_squared));
				}
				else
				{
					scratch0_xxxx = scratch0[0];
					scratch0_yyyy = scratch0[1];
					scratch0_zzzz = scratch0[2];
					scratch0_wwww = scratch0[3];

					scratch1_xxxx = scratch1[0];
					scratch1_yyyy = scratch1[1];
					scratch1_zzzz = scratch1[2];
					scratch1_wwww = scratch1[3];
				}

				// Interpolate linearly and store our rotations in SOA
				{
					// Calculate the vector4 dot product: dot(start, end)
					const rtm::vector4f xxxx_squared = rtm::vector_mul(scratch0_xxxx, scratch1_xxxx);
					const rtm::vector4f yyyy_squared = rtm::vector_mul(scratch0_yyyy, scratch1_yyyy);
					const rtm::vector4f zzzz_squared = rtm::vector_mul(scratch0_zzzz, scratch1_zzzz);
					const rtm::vector4f wwww_squared = rtm::vector_mul(scratch0_wwww, scratch1_wwww);

					const rtm::vector4f dot4 = rtm::vector_add(rtm::vector_add(rtm::vector_add(xxxx_squared, yyyy_squared), zzzz_squared), wwww_squared);

					// Calculate the bias, if the dot product is positive or zero, there is no bias
					// but if it is negative, we want to flip the 'end' rotation XYZW components
					const rtm::vector4f neg_zero = rtm::vector_set(-0.0F);
					const rtm::vector4f bias = vector_and(dot4, neg_zero);

					// Apply our bias to the 'end'
					scratch1_xxxx = vector_xor(scratch1_xxxx, bias);
					scratch1_yyyy = vector_xor(scratch1_yyyy, bias);
					scratch1_zzzz = vector_xor(scratch1_zzzz, bias);
					scratch1_wwww = vector_xor(scratch1_wwww, bias);

					// Lerp the rotation after applying the bias
					// ((1.0 - alpha) * start) + (alpha * (end ^ bias)) == (start - alpha * start) + (alpha * (end ^ bias))
					const rtm::vector4f alpha = rtm::vector_set(decomp_context.interpolation_alpha);

					rtm::vector4f interp_xxxx = rtm::vector_mul_add(scratch1_xxxx, alpha, rtm::vector_neg_mul_sub(scratch0_xxxx, alpha, scratch0_xxxx));
					rtm::vector4f interp_yyyy = rtm::vector_mul_add(scratch1_yyyy, alpha, rtm::vector_neg_mul_sub(scratch0_yyyy, alpha, scratch0_yyyy));
					rtm::vector4f interp_zzzz = rtm::vector_mul_add(scratch1_zzzz, alpha, rtm::vector_neg_mul_sub(scratch0_zzzz, alpha, scratch0_zzzz));
					rtm::vector4f interp_wwww = rtm::vector_mul_add(scratch1_wwww, alpha, rtm::vector_neg_mul_sub(scratch0_wwww, alpha, scratch0_wwww));

					// Due to the interpolation, the result might not be anywhere near normalized!
					// Make sure to normalize afterwards before using
					const bool normalize_rotations = decompression_settings_type::normalize_rotations();
					if (normalize_rotations)
					{
						const rtm::vector4f interp_xxxx_squared = rtm::vector_mul(interp_xxxx, interp_xxxx);
						const rtm::vector4f interp_yyyy_squared = rtm::vector_mul(interp_yyyy, interp_yyyy);
						const rtm::vector4f interp_zzzz_squared = rtm::vector_mul(interp_zzzz, interp_zzzz);
						const rtm::vector4f interp_wwww_squared = rtm::vector_mul(interp_wwww, interp_wwww);

						const rtm::vector4f interp_dot4 = rtm::vector_add(rtm::vector_add(rtm::vector_add(interp_xxxx_squared, interp_yyyy_squared), interp_zzzz_squared), interp_wwww_squared);

						const rtm::vector4f interp_len = rtm::vector_sqrt(interp_dot4);
						const rtm::vector4f interp_inv_len = rtm::vector_div(rtm::vector_set(1.0F), interp_len);

						interp_xxxx = rtm::vector_mul(interp_xxxx, interp_inv_len);
						interp_yyyy = rtm::vector_mul(interp_yyyy, interp_inv_len);
						interp_zzzz = rtm::vector_mul(interp_zzzz, interp_inv_len);
						interp_wwww = rtm::vector_mul(interp_wwww, interp_inv_len);
					}

					// Swizzle out our 4 samples
					const rtm::vector4f tmp0 = rtm::vector_mix<rtm::mix4::x, rtm::mix4::y, rtm::mix4::a, rtm::mix4::b>(interp_xxxx, interp_yyyy);
					const rtm::vector4f tmp1 = rtm::vector_mix<rtm::mix4::z, rtm::mix4::w, rtm::mix4::c, rtm::mix4::d>(interp_xxxx, interp_yyyy);
					const rtm::vector4f tmp2 = rtm::vector_mix<rtm::mix4::x, rtm::mix4::y, rtm::mix4::a, rtm::mix4::b>(interp_zzzz, interp_wwww);
					const rtm::vector4f tmp3 = rtm::vector_mix<rtm::mix4::z, rtm::mix4::w, rtm::mix4::c, rtm::mix4::d>(interp_zzzz, interp_wwww);

					const rtm::vector4f sample0 = rtm::vector_mix<rtm::mix4::x, rtm::mix4::z, rtm::mix4::a, rtm::mix4::c>(tmp0, tmp2);
					const rtm::vector4f sample1 = rtm::vector_mix<rtm::mix4::y, rtm::mix4::w, rtm::mix4::b, rtm::mix4::d>(tmp0, tmp2);
					const rtm::vector4f sample2 = rtm::vector_mix<rtm::mix4::x, rtm::mix4::z, rtm::mix4::a, rtm::mix4::c>(tmp1, tmp3);
					const rtm::vector4f sample3 = rtm::vector_mix<rtm::mix4::y, rtm::mix4::w, rtm::mix4::b, rtm::mix4::d>(tmp1, tmp3);

					rotations.cached_samples[cache_write_index + 0] = sample0;
					rotations.cached_samples[cache_write_index + 1] = sample1;
					rotations.cached_samples[cache_write_index + 2] = sample2;
					rotations.cached_samples[cache_write_index + 3] = sample3;
					cache_write_index += 4;
				}
			}

			rtm::quatf RTM_SIMD_CALL consume_rotation()
			{
				ACL_ASSERT(rotations.cache_read_index < rotations.cache_write_index, "Attempting to consume an animated sample that isn't cached");
				const uint32_t cache_read_index = rotations.cache_read_index++;
				return rotations.cached_samples[cache_read_index % 8];
			}

			template<class decompression_settings_adapter_type>
			void unpack_translations(const persistent_transform_decompression_context_v0& decomp_context)
			{
				uint32_t num_left_to_unpack = translations.num_left_to_unpack;
				if (num_left_to_unpack == 0)
					return;	// Nothing left to do, we are done

				// If we have less than 4 cached samples, unpack 4 more and prefetch the next cache line
				const uint32_t num_cached = translations.get_num_cached();
				if (num_cached >= 4)
					return;	// Enough cached, nothing to do

				const uint32_t num_to_unpack = std::min<uint32_t>(num_left_to_unpack, 4);
				num_left_to_unpack -= num_to_unpack;
				translations.num_left_to_unpack = num_left_to_unpack;

				// Write index will be either 0 or 4 here since we always unpack 4 at a time
				uint32_t cache_write_index = translations.cache_write_index % 8;
				translations.cache_write_index += num_to_unpack;

				unpack_animated_vector3<decompression_settings_adapter_type>(decomp_context, scratch0, num_to_unpack, clip_sampling_context, segment_sampling_context[0]);
				unpack_animated_vector3<decompression_settings_adapter_type>(decomp_context, scratch1, num_to_unpack, clip_sampling_context, segment_sampling_context[1]);

				const float interpolation_alpha = decomp_context.interpolation_alpha;
				for (uint32_t unpack_index = 0; unpack_index < num_to_unpack; ++unpack_index)
				{
					const rtm::vector4f sample0 = scratch0[unpack_index];
					const rtm::vector4f sample1 = scratch1[unpack_index];

					const rtm::quatf sample = rtm::vector_lerp(sample0, sample1, interpolation_alpha);

					translations.cached_samples[cache_write_index] = sample;
					cache_write_index++;
				}

				// If we have some range reduction, skip the data we read
				if (are_any_enum_flags_set(decomp_context.range_reduction, range_reduction_flags8::translations))
				{
					const uint32_t range_entry_size = 3 * sizeof(float);
					clip_sampling_context.clip_range_data += num_to_unpack * range_entry_size * 2;
				}

				// Clip range data is 24 bytes per sub-track and as such we need to prefetch two cache lines ahead to process 4 sub-tracks
				ACL_IMPL_ANIMATED_PREFETCH(clip_sampling_context.clip_range_data + 63);
				ACL_IMPL_ANIMATED_PREFETCH(clip_sampling_context.clip_range_data + 127);
			}

			rtm::vector4f RTM_SIMD_CALL consume_translation()
			{
				ACL_ASSERT(translations.cache_read_index < translations.cache_write_index, "Attempting to consume an animated sample that isn't cached");
				const uint32_t cache_read_index = translations.cache_read_index++;
				return translations.cached_samples[cache_read_index % 8];
			}

			template<class decompression_settings_adapter_type>
			void unpack_scales(const persistent_transform_decompression_context_v0& decomp_context)
			{
				uint32_t num_left_to_unpack = scales.num_left_to_unpack;
				if (num_left_to_unpack == 0)
					return;	// Nothing left to do, we are done

				// If we have less than 4 cached samples, unpack 4 more and prefetch the next cache line
				const uint32_t num_cached = scales.get_num_cached();
				if (num_cached >= 4)
					return;	// Enough cached, nothing to do

				const uint32_t num_to_unpack = std::min<uint32_t>(num_left_to_unpack, 4);
				num_left_to_unpack -= num_to_unpack;
				scales.num_left_to_unpack = num_left_to_unpack;

				// Write index will be either 0 or 4 here since we always unpack 4 at a time
				uint32_t cache_write_index = scales.cache_write_index % 8;
				scales.cache_write_index += num_to_unpack;

				unpack_animated_vector3<decompression_settings_adapter_type>(decomp_context, scratch0, num_to_unpack, clip_sampling_context, segment_sampling_context[0]);
				unpack_animated_vector3<decompression_settings_adapter_type>(decomp_context, scratch1, num_to_unpack, clip_sampling_context, segment_sampling_context[1]);

				const float interpolation_alpha = decomp_context.interpolation_alpha;
				for (uint32_t unpack_index = 0; unpack_index < num_to_unpack; ++unpack_index)
				{
					const rtm::vector4f sample0 = scratch0[unpack_index];
					const rtm::vector4f sample1 = scratch1[unpack_index];

					const rtm::quatf sample = rtm::vector_lerp(sample0, sample1, interpolation_alpha);

					scales.cached_samples[cache_write_index] = sample;
					cache_write_index++;
				}

				// If we have some range reduction, skip the data we read
				if (are_any_enum_flags_set(decomp_context.range_reduction, range_reduction_flags8::scales))
				{
					const uint32_t range_entry_size = 3 * sizeof(float);
					clip_sampling_context.clip_range_data += num_to_unpack * range_entry_size * 2;
				}

				// Clip range data is 24 bytes per sub-track and as such we need to prefetch two cache lines ahead to process 4 sub-tracks
				ACL_IMPL_ANIMATED_PREFETCH(clip_sampling_context.clip_range_data + 63);
				ACL_IMPL_ANIMATED_PREFETCH(clip_sampling_context.clip_range_data + 127);
			}

			rtm::vector4f RTM_SIMD_CALL consume_scale()
			{
				ACL_ASSERT(scales.cache_read_index < scales.cache_write_index, "Attempting to consume an animated sample that isn't cached");
				const uint32_t cache_read_index = scales.cache_read_index++;
				return scales.cached_samples[cache_read_index % 8];
			}
		};

		// TODO: Stage bitset decomp
		// TODO: Merge the per track format and segment range info into a single buffer? Less to prefetch and used together
		// TODO: How do we hide the cache miss after the seek to read the segment header? What work can we do while we prefetch?
		// TODO: Swizzle rotation clip range data for SOA
		// TODO: Port vector3 decomp to use SOA
		// TODO: Unroll quat unpacking and convert to SOA
		// TODO: Use AVX where we can
		// TODO: Implement optimized NEON transpose with:
		//    float32x4x2_t A = vzipq_f32(xxxx, zzzz);
		//    float32x4x2_t B = vzipq_f32(yyyy, wwww);
		//    float32x4x2_t C = vzipq_f32(A.val[0], B.val[0]);
		//    float32x4x2_t D = vzipq_f32(A.val[1], B.val[1]);
		//    xxxx = C.val[0]; yyyy = C.val[1]; zzzz = D.val[0]; wwww = D.val[1]

		template<class decompression_settings_type, class track_writer_type>
		inline void decompress_tracks_v0(const persistent_transform_decompression_context_v0& context, track_writer_type& writer)
		{
			ACL_ASSERT(context.sample_time >= 0.0f, "Context not set to a valid sample time");
			if (context.sample_time < 0.0F)
				return;	// Invalid sample time, we didn't seek yet

			// Due to the SIMD operations, we sometimes overflow in the SIMD lanes not used.
			// Disable floating point exceptions to avoid issues.
			fp_environment fp_env;
			if (decompression_settings_type::disable_fp_exeptions())
				disable_fp_exceptions(fp_env);

			const tracks_header& header = get_tracks_header(*context.tracks);

			using translation_adapter = acl_impl::translation_decompression_settings_adapter<decompression_settings_type>;
			using scale_adapter = acl_impl::scale_decompression_settings_adapter<decompression_settings_type>;

			const rtm::vector4f default_translation = rtm::vector_zero();
			const rtm::vector4f default_scale = rtm::vector_set(float(header.get_default_scale()));
			const bool has_scale = header.get_has_scale();
			const uint32_t num_tracks = header.num_tracks;

			sampling_context_v0 sampling_context_;
			sampling_context_.track_index = 0;
			sampling_context_.constant_track_data_offset = 0;
			sampling_context_.clip_range_data_offset = 0;
			sampling_context_.format_per_track_data_offset = 0;
			sampling_context_.segment_range_data_offset = 0;
			sampling_context_.key_frame_bit_offsets[0] = context.key_frame_bit_offsets[0];
			sampling_context_.key_frame_bit_offsets[1] = context.key_frame_bit_offsets[1];

			sampling_context_.vectors[0] = default_translation;	// Init with something to avoid GCC warning
			sampling_context_.vectors[1] = default_translation;	// Init with something to avoid GCC warning

#if defined(ACL_IMPL_USE_STAGED_CONSTANT_DECOMPRESSION)
			constant_track_cache_v0 constant_track_cache;
			constant_track_cache.initialize<decompression_settings_type>(context);

#if defined(ACL_IMPL_USE_STAGED_ANIMATED_DECOMPRESSION)
			animated_track_cache_v0 animated_track_cache;
			animated_track_cache.initialize(context);
#endif

			uint32_t sub_track_index = 0;

			for (uint32_t track_index = 0; track_index < num_tracks; ++track_index)
			{
				if ((track_index % 4) == 0)
				{
					constant_track_cache.unpack_rotations<decompression_settings_type>(context);
					constant_track_cache.unpack_translations();

					if (has_scale)
						constant_track_cache.unpack_scales();

#if defined(ACL_IMPL_USE_STAGED_ANIMATED_DECOMPRESSION)
					animated_track_cache.unpack_rotations<decompression_settings_type>(context);
					animated_track_cache.unpack_translations<translation_adapter>(context);

					if (has_scale)
						animated_track_cache.unpack_scales<scale_adapter>(context);
#endif
				}

				{
					const bitset_index_ref track_index_bit_ref(context.bitset_desc, sub_track_index);
					const bool is_sample_default = bitset_test(context.default_tracks_bitset, track_index_bit_ref);
					rtm::quatf rotation;
					if (is_sample_default)
					{
						rotation = rtm::quat_identity();
					}
					else
					{
						const bool is_sample_constant = bitset_test(context.constant_tracks_bitset, track_index_bit_ref);
						if (is_sample_constant)
							rotation = constant_track_cache.consume_rotation();
						else
#if defined(ACL_IMPL_USE_STAGED_ANIMATED_DECOMPRESSION)
							rotation = animated_track_cache.consume_rotation();
#else
							rotation = decompress_and_interpolate_animated_rotation<decompression_settings_type>(context, sampling_context_);
#endif
					}

					ACL_ASSERT(rtm::quat_is_finite(rotation), "Rotation is not valid!");
					ACL_ASSERT(rtm::quat_is_normalized(rotation), "Rotation is not normalized!");

					writer.write_rotation(track_index, rotation);
					sub_track_index++;
				}

				{
					const bitset_index_ref track_index_bit_ref(context.bitset_desc, sub_track_index);
					const bool is_sample_default = bitset_test(context.default_tracks_bitset, track_index_bit_ref);
					rtm::vector4f translation;
					if (is_sample_default)
					{
						translation = default_translation;
					}
					else
					{
						const bool is_sample_constant = bitset_test(context.constant_tracks_bitset, track_index_bit_ref);
						if (is_sample_constant)
							translation = constant_track_cache.consume_translation();
						else
#if defined(ACL_IMPL_USE_STAGED_ANIMATED_DECOMPRESSION)
							translation = animated_track_cache.consume_translation();
#else
							translation = decompress_and_interpolate_animated_vector3<translation_adapter>(context, sampling_context_);
#endif
					}

					ACL_ASSERT(rtm::vector_is_finite3(translation), "Translation is not valid!");

					writer.write_translation(track_index, translation);
					sub_track_index++;
				}

				if (has_scale)
				{
					const bitset_index_ref track_index_bit_ref(context.bitset_desc, sub_track_index);
					const bool is_sample_default = bitset_test(context.default_tracks_bitset, track_index_bit_ref);
					rtm::vector4f scale;
					if (is_sample_default)
					{
						scale = default_scale;
					}
					else
					{
						const bool is_sample_constant = bitset_test(context.constant_tracks_bitset, track_index_bit_ref);
						if (is_sample_constant)
							scale = constant_track_cache.consume_scale();
						else
#if defined(ACL_IMPL_USE_STAGED_ANIMATED_DECOMPRESSION)
							scale = animated_track_cache.consume_scale();
#else
							scale = decompress_and_interpolate_animated_vector3<scale_adapter>(context, sampling_context_);
#endif
					}

					ACL_ASSERT(rtm::vector_is_finite3(scale), "Scale is not valid!");

					writer.write_scale(track_index, scale);
					sub_track_index++;
				}
				else
					writer.write_scale(track_index, default_scale);
			}
#else
			for (uint32_t track_index = 0; track_index < num_tracks; ++track_index)
			{
				if (track_writer_type::skip_all_rotations() || writer.skip_track_rotation(track_index))
					skip_over_rotation<decompression_settings_type>(context, sampling_context_);
				else
				{
					const rtm::quatf rotation = decompress_and_interpolate_rotation<decompression_settings_type>(context, sampling_context_);
					writer.write_rotation(track_index, rotation);
				}

				if (track_writer_type::skip_all_translations() || writer.skip_track_translation(track_index))
					skip_over_vector<translation_adapter>(context, sampling_context_);
				else
				{
					const rtm::vector4f translation = decompress_and_interpolate_vector<translation_adapter>(context, default_translation, sampling_context_);
					writer.write_translation(track_index, translation);
				}

				if (track_writer_type::skip_all_scales() || writer.skip_track_scale(track_index))
				{
					if (has_scale)
						skip_over_vector<scale_adapter>(context, sampling_context_);
				}
				else
				{
					const rtm::vector4f scale = has_scale ? decompress_and_interpolate_vector<scale_adapter>(context, default_scale, sampling_context_) : default_scale;
					writer.write_scale(track_index, scale);
				}
			}
#endif

			if (decompression_settings_type::disable_fp_exeptions())
				restore_fp_exceptions(fp_env);
		}

		template<class decompression_settings_type, class track_writer_type>
		inline void decompress_track_v0(const persistent_transform_decompression_context_v0& context, uint32_t track_index, track_writer_type& writer)
		{
			ACL_ASSERT(context.sample_time >= 0.0f, "Context not set to a valid sample time");
			if (context.sample_time < 0.0F)
				return;	// Invalid sample time, we didn't seek yet

			const tracks_header& tracks_header_ = get_tracks_header(*context.tracks);
			ACL_ASSERT(track_index < tracks_header_.num_tracks, "Invalid track index");

			if (track_index >= tracks_header_.num_tracks)
				return;	// Invalid track index

			// Due to the SIMD operations, we sometimes overflow in the SIMD lanes not used.
			// Disable floating point exceptions to avoid issues.
			fp_environment fp_env;
			if (decompression_settings_type::disable_fp_exeptions())
				disable_fp_exceptions(fp_env);

			using translation_adapter = acl_impl::translation_decompression_settings_adapter<decompression_settings_type>;
			using scale_adapter = acl_impl::scale_decompression_settings_adapter<decompression_settings_type>;

			const rtm::vector4f default_translation = rtm::vector_zero();
			const rtm::vector4f default_scale = rtm::vector_set(float(tracks_header_.get_default_scale()));
			const bool has_scale = tracks_header_.get_has_scale();

			sampling_context_v0 sampling_context_;

#if defined(ACL_IMPL_USE_STAGED_CONSTANT_DECOMPRESSION)
			sampling_context_.track_index = 0;
			sampling_context_.constant_track_data_offset = 0;
			sampling_context_.clip_range_data_offset = 0;
			sampling_context_.format_per_track_data_offset = 0;
			sampling_context_.segment_range_data_offset = 0;
			sampling_context_.key_frame_bit_offsets[0] = context.key_frame_bit_offsets[0];
			sampling_context_.key_frame_bit_offsets[1] = context.key_frame_bit_offsets[1];

			sampling_context_.vectors[0] = default_translation;	// Init with something to avoid GCC warning
			sampling_context_.vectors[1] = default_translation;	// Init with something to avoid GCC warning

			constant_track_cache_v0 constant_track_cache;
			constant_track_cache.initialize<decompression_settings_type>(context);

			// Unpack our first batch, this will stall on a cache miss and prefetch the next batch
			constant_track_cache.unpack_rotations<decompression_settings_type>(context);
			constant_track_cache.unpack_translations();

			if (has_scale)
				constant_track_cache.unpack_scales();

#if defined(ACL_IMPL_USE_STAGED_ANIMATED_DECOMPRESSION)
			animated_track_cache_v0 animated_track_cache;
			animated_track_cache.initialize(context);

			// Unpack our first batch, this will stall on a cache miss and prefetch the next batch
			animated_track_cache.unpack_rotations<decompression_settings_type>(context);
			animated_track_cache.unpack_translations<translation_adapter>(context);

			if (has_scale)
				animated_track_cache.unpack_scales<scale_adapter>(context);
#endif

			uint32_t sub_track_index = 0;
			for (uint32_t skipped_track_index = 0; skipped_track_index < track_index; ++skipped_track_index)
			{
				// Skip our track
				{
					const bitset_index_ref track_index_bit_ref(context.bitset_desc, sub_track_index);
					const bool is_sample_default = bitset_test(context.default_tracks_bitset, track_index_bit_ref);
					if (!is_sample_default)
					{
						const bool is_sample_constant = bitset_test(context.constant_tracks_bitset, track_index_bit_ref);
						if (is_sample_constant)
							constant_track_cache.consume_rotation();
						else
#if defined(ACL_IMPL_USE_STAGED_ANIMATED_DECOMPRESSION)
							animated_track_cache.consume_rotation();
#else
							decompress_and_interpolate_animated_rotation<decompression_settings_type>(context, sampling_context_);
#endif
					}

					sub_track_index++;
				}

				{
					const bitset_index_ref track_index_bit_ref(context.bitset_desc, sub_track_index);
					const bool is_sample_default = bitset_test(context.default_tracks_bitset, track_index_bit_ref);
					if (!is_sample_default)
					{
						const bool is_sample_constant = bitset_test(context.constant_tracks_bitset, track_index_bit_ref);
						if (is_sample_constant)
							constant_track_cache.consume_translation();
						else
#if defined(ACL_IMPL_USE_STAGED_ANIMATED_DECOMPRESSION)
							animated_track_cache.consume_translation();
#else
							decompress_and_interpolate_animated_vector3<translation_adapter>(context, sampling_context_);
#endif
					}

					sub_track_index++;
				}

				if (has_scale)
				{
					const bitset_index_ref track_index_bit_ref(context.bitset_desc, sub_track_index);
					const bool is_sample_default = bitset_test(context.default_tracks_bitset, track_index_bit_ref);
					if (!is_sample_default)
					{
						const bool is_sample_constant = bitset_test(context.constant_tracks_bitset, track_index_bit_ref);
						if (is_sample_constant)
							constant_track_cache.consume_scale();
						else
#if defined(ACL_IMPL_USE_STAGED_ANIMATED_DECOMPRESSION)
							animated_track_cache.consume_scale();
#else
							decompress_and_interpolate_animated_vector3<scale_adapter>(context, sampling_context_);
#endif
					}

					sub_track_index++;
				}

				// Every 4th track, attempt to unpack the next 4
				if ((skipped_track_index % 4) == 3)
				{
					constant_track_cache.unpack_rotations<decompression_settings_type>(context);
					constant_track_cache.unpack_translations();

					if (has_scale)
						constant_track_cache.unpack_scales();

#if defined(ACL_IMPL_USE_STAGED_ANIMATED_DECOMPRESSION)
					// Unpack our first batch, this will stall on a cache miss and prefetch the next batch
					animated_track_cache.unpack_rotations<decompression_settings_type>(context);
					animated_track_cache.unpack_translations<translation_adapter>(context);

					if (has_scale)
						animated_track_cache.unpack_scales<scale_adapter>(context);
#endif
				}
			}

			// Finally reached our desired track, unpack it

			{
				const bitset_index_ref track_index_bit_ref(context.bitset_desc, sub_track_index);
				const bool is_sample_default = bitset_test(context.default_tracks_bitset, track_index_bit_ref);
				rtm::quatf rotation;
				if (is_sample_default)
				{
					rotation = rtm::quat_identity();
				}
				else
				{
					const bool is_sample_constant = bitset_test(context.constant_tracks_bitset, track_index_bit_ref);
					if (is_sample_constant)
						rotation = constant_track_cache.consume_rotation();
					else
#if defined(ACL_IMPL_USE_STAGED_ANIMATED_DECOMPRESSION)
						rotation = animated_track_cache.consume_rotation();
#else
						rotation = decompress_and_interpolate_animated_rotation<decompression_settings_type>(context, sampling_context_);
#endif
				}

				writer.write_rotation(track_index, rotation);
				sub_track_index++;
			}

			{
				const bitset_index_ref track_index_bit_ref(context.bitset_desc, sub_track_index);
				const bool is_sample_default = bitset_test(context.default_tracks_bitset, track_index_bit_ref);
				rtm::vector4f translation;
				if (is_sample_default)
				{
					translation = default_translation;
				}
				else
				{
					const bool is_sample_constant = bitset_test(context.constant_tracks_bitset, track_index_bit_ref);
					if (is_sample_constant)
						translation = constant_track_cache.consume_translation();
					else
#if defined(ACL_IMPL_USE_STAGED_ANIMATED_DECOMPRESSION)
						translation = animated_track_cache.consume_translation();
#else
						translation = decompress_and_interpolate_animated_vector3<translation_adapter>(context, sampling_context_);
#endif
				}

				writer.write_translation(track_index, translation);
				sub_track_index++;
			}

			if (has_scale)
			{
				const bitset_index_ref track_index_bit_ref(context.bitset_desc, sub_track_index);
				const bool is_sample_default = bitset_test(context.default_tracks_bitset, track_index_bit_ref);
				rtm::vector4f scale;
				if (is_sample_default)
				{
					scale = default_scale;
				}
				else
				{
					const bool is_sample_constant = bitset_test(context.constant_tracks_bitset, track_index_bit_ref);
					if (is_sample_constant)
						scale = constant_track_cache.consume_scale();
					else
#if defined(ACL_IMPL_USE_STAGED_ANIMATED_DECOMPRESSION)
						scale = animated_track_cache.consume_scale();
#else
						scale = decompress_and_interpolate_animated_vector3<scale_adapter>(context, sampling_context_);
#endif
				}

				writer.write_scale(track_index, scale);
			}
			else
				writer.write_scale(track_index, default_scale);
#else
			sampling_context_.key_frame_bit_offsets[0] = context.key_frame_bit_offsets[0];
			sampling_context_.key_frame_bit_offsets[1] = context.key_frame_bit_offsets[1];

			const rotation_format8 rotation_format = get_rotation_format<decompression_settings_type>(context.rotation_format);
			const vector_format8 translation_format = get_vector_format<translation_adapter>(context.translation_format);
			const vector_format8 scale_format = get_vector_format<scale_adapter>(context.scale_format);

			const bool are_all_tracks_variable = is_rotation_format_variable(rotation_format) && is_vector_format_variable(translation_format) && is_vector_format_variable(scale_format);
			if (!are_all_tracks_variable)
			{
				// Slow path, not optimized yet because it's more complex and shouldn't be used in production anyway
				sampling_context_.track_index = 0;
				sampling_context_.constant_track_data_offset = 0;
				sampling_context_.clip_range_data_offset = 0;
				sampling_context_.format_per_track_data_offset = 0;
				sampling_context_.segment_range_data_offset = 0;

				for (uint32_t bone_index = 0; bone_index < track_index; ++bone_index)
				{
					skip_over_rotation<decompression_settings_type>(context, sampling_context_);
					skip_over_vector<translation_adapter>(context, sampling_context_);

					if (has_scale)
						skip_over_vector<scale_adapter>(context, sampling_context_);
				}
			}
			else
			{
				const uint32_t num_tracks_per_bone = has_scale ? 3 : 2;
				const uint32_t sub_track_index = track_index * num_tracks_per_bone;
				uint32_t num_default_rotations = 0;
				uint32_t num_default_translations = 0;
				uint32_t num_default_scales = 0;
				uint32_t num_constant_rotations = 0;
				uint32_t num_constant_translations = 0;
				uint32_t num_constant_scales = 0;

				if (has_scale)
				{
					uint32_t rotation_track_bit_mask = 0x92492492;		// b100100100..
					uint32_t translation_track_bit_mask = 0x49249249;	// b010010010..
					uint32_t scale_track_bit_mask = 0x24924924;			// b001001001..

					const uint32_t last_offset = sub_track_index / 32;
					uint32_t offset = 0;
					for (; offset < last_offset; ++offset)
					{
						const uint32_t default_value = context.default_tracks_bitset[offset];
						num_default_rotations += count_set_bits(default_value & rotation_track_bit_mask);
						num_default_translations += count_set_bits(default_value & translation_track_bit_mask);
						num_default_scales += count_set_bits(default_value & scale_track_bit_mask);

						const uint32_t constant_value = context.constant_tracks_bitset[offset];
						num_constant_rotations += count_set_bits(constant_value & rotation_track_bit_mask);
						num_constant_translations += count_set_bits(constant_value & translation_track_bit_mask);
						num_constant_scales += count_set_bits(constant_value & scale_track_bit_mask);

						// Because the number of tracks in a 32 bit value isn't a multiple of the number of tracks we have (3),
						// we have to cycle the masks. There are 3 possible masks, just swap them.
						const uint32_t old_rotation_track_bit_mask = rotation_track_bit_mask;
						rotation_track_bit_mask = translation_track_bit_mask;
						translation_track_bit_mask = scale_track_bit_mask;
						scale_track_bit_mask = old_rotation_track_bit_mask;
					}

					const uint32_t remaining_tracks = sub_track_index % 32;
					if (remaining_tracks != 0)
					{
						const uint32_t not_up_to_track_mask = ((1 << (32 - remaining_tracks)) - 1);
						const uint32_t default_value = and_not(not_up_to_track_mask, context.default_tracks_bitset[offset]);
						num_default_rotations += count_set_bits(default_value & rotation_track_bit_mask);
						num_default_translations += count_set_bits(default_value & translation_track_bit_mask);
						num_default_scales += count_set_bits(default_value & scale_track_bit_mask);

						const uint32_t constant_value = and_not(not_up_to_track_mask, context.constant_tracks_bitset[offset]);
						num_constant_rotations += count_set_bits(constant_value & rotation_track_bit_mask);
						num_constant_translations += count_set_bits(constant_value & translation_track_bit_mask);
						num_constant_scales += count_set_bits(constant_value & scale_track_bit_mask);
					}
				}
				else
				{
					const uint32_t rotation_track_bit_mask = 0xAAAAAAAA;		// b10101010..
					const uint32_t translation_track_bit_mask = 0x55555555;		// b01010101..

					const uint32_t last_offset = sub_track_index / 32;
					uint32_t offset = 0;
					for (; offset < last_offset; ++offset)
					{
						const uint32_t default_value = context.default_tracks_bitset[offset];
						num_default_rotations += count_set_bits(default_value & rotation_track_bit_mask);
						num_default_translations += count_set_bits(default_value & translation_track_bit_mask);

						const uint32_t constant_value = context.constant_tracks_bitset[offset];
						num_constant_rotations += count_set_bits(constant_value & rotation_track_bit_mask);
						num_constant_translations += count_set_bits(constant_value & translation_track_bit_mask);
					}

					const uint32_t remaining_tracks = sub_track_index % 32;
					if (remaining_tracks != 0)
					{
						const uint32_t not_up_to_track_mask = ((1 << (32 - remaining_tracks)) - 1);
						const uint32_t default_value = and_not(not_up_to_track_mask, context.default_tracks_bitset[offset]);
						num_default_rotations += count_set_bits(default_value & rotation_track_bit_mask);
						num_default_translations += count_set_bits(default_value & translation_track_bit_mask);

						const uint32_t constant_value = and_not(not_up_to_track_mask, context.constant_tracks_bitset[offset]);
						num_constant_rotations += count_set_bits(constant_value & rotation_track_bit_mask);
						num_constant_translations += count_set_bits(constant_value & translation_track_bit_mask);
					}
				}

				// Tracks that are default are also constant
				const uint32_t num_animated_rotations = track_index - num_constant_rotations;
				const uint32_t num_animated_translations = track_index - num_constant_translations;

				const rotation_format8 packed_rotation_format = is_rotation_format_variable(rotation_format) ? get_highest_variant_precision(get_rotation_variant(rotation_format)) : rotation_format;
				const uint32_t packed_rotation_size = get_packed_rotation_size(packed_rotation_format);

				uint32_t constant_track_data_offset = (num_constant_rotations - num_default_rotations) * packed_rotation_size;
				constant_track_data_offset += (num_constant_translations - num_default_translations) * get_packed_vector_size(vector_format8::vector3f_full);

				uint32_t clip_range_data_offset = 0;
				uint32_t segment_range_data_offset = 0;

				const range_reduction_flags8 range_reduction = context.range_reduction;
				if (are_any_enum_flags_set(range_reduction, range_reduction_flags8::rotations))
				{
					clip_range_data_offset += context.num_rotation_components * sizeof(float) * 2 * num_animated_rotations;

					if (context.has_segments)
						segment_range_data_offset += context.num_rotation_components * k_segment_range_reduction_num_bytes_per_component * 2 * num_animated_rotations;
				}

				if (are_any_enum_flags_set(range_reduction, range_reduction_flags8::translations))
				{
					clip_range_data_offset += k_clip_range_reduction_vector3_range_size * num_animated_translations;

					if (context.has_segments)
						segment_range_data_offset += 3 * k_segment_range_reduction_num_bytes_per_component * 2 * num_animated_translations;
				}

				uint32_t num_animated_tracks = num_animated_rotations + num_animated_translations;
				if (has_scale)
				{
					const uint32_t num_animated_scales = track_index - num_constant_scales;
					num_animated_tracks += num_animated_scales;

					constant_track_data_offset += (num_constant_scales - num_default_scales) * get_packed_vector_size(vector_format8::vector3f_full);

					if (are_any_enum_flags_set(range_reduction, range_reduction_flags8::scales))
					{
						clip_range_data_offset += k_clip_range_reduction_vector3_range_size * num_animated_scales;

						if (context.has_segments)
							segment_range_data_offset += 3 * k_segment_range_reduction_num_bytes_per_component * 2 * num_animated_scales;
					}
				}

				sampling_context_.track_index = sub_track_index;
				sampling_context_.constant_track_data_offset = constant_track_data_offset;
				sampling_context_.clip_range_data_offset = clip_range_data_offset;
				sampling_context_.segment_range_data_offset = segment_range_data_offset;
				sampling_context_.format_per_track_data_offset = num_animated_tracks;

				for (uint32_t animated_track_index = 0; animated_track_index < num_animated_tracks; ++animated_track_index)
				{
					const uint8_t bit_rate0 = context.format_per_track_data[0][animated_track_index];
					const uint32_t num_bits_at_bit_rate0 = get_num_bits_at_bit_rate(bit_rate0) * 3;	// 3 components

					sampling_context_.key_frame_bit_offsets[0] += num_bits_at_bit_rate0;

					const uint8_t bit_rate1 = context.format_per_track_data[1][animated_track_index];
					const uint32_t num_bits_at_bit_rate1 = get_num_bits_at_bit_rate(bit_rate1) * 3;	// 3 components

					sampling_context_.key_frame_bit_offsets[1] += num_bits_at_bit_rate1;
				}
			}

			sampling_context_.vectors[0] = default_translation;	// Init with something to avoid GCC warning
			sampling_context_.vectors[1] = default_translation;	// Init with something to avoid GCC warning

			const rtm::quatf rotation = decompress_and_interpolate_rotation<decompression_settings_type>(context, sampling_context_);
			writer.write_rotation(track_index, rotation);

			const rtm::vector4f translation = decompress_and_interpolate_vector<translation_adapter>(context, default_translation, sampling_context_);
			writer.write_translation(track_index, translation);

			const rtm::vector4f scale = has_scale ? decompress_and_interpolate_vector<scale_adapter>(context, default_scale, sampling_context_) : default_scale;
			writer.write_scale(track_index, scale);
#endif

			if (decompression_settings_type::disable_fp_exeptions())
				restore_fp_exceptions(fp_env);
		}
	}
}

ACL_IMPL_FILE_PRAGMA_POP
