#pragma once

////////////////////////////////////////////////////////////////////////////////
// The MIT License (MIT)
//
// Copyright (c) 2017 Nicholas Frechette & Animation Compression Library contributors
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

#include "acl/core/iallocator.h"
#include "acl/core/impl/compiler_utils.h"
#include "acl/core/error.h"
#include "acl/core/enum_utils.h"
#include "acl/core/track_formats.h"
#include "acl/core/track_types.h"
#include "acl/core/range_reduction_types.h"
#include "acl/core/variable_bit_rates.h"
#include "acl/math/quat_packing.h"
#include "acl/math/vector4_packing.h"
#include "acl/compression/impl/animated_track_utils.h"
#include "acl/compression/impl/clip_context.h"

#include "acl/core/impl/compressed_headers.h"

#include <rtm/vector4f.h>

#include <cstdint>

ACL_IMPL_FILE_PRAGMA_PUSH

namespace acl
{
	namespace acl_impl
	{
		inline uint32_t get_stream_range_data_size(const clip_context& clip, range_reduction_flags8 range_reduction, rotation_format8 rotation_format)
		{
			const uint32_t rotation_size = are_any_enum_flags_set(range_reduction, range_reduction_flags8::rotations) ? get_range_reduction_rotation_size(rotation_format) : 0;
			const uint32_t translation_size = are_any_enum_flags_set(range_reduction, range_reduction_flags8::translations) ? k_clip_range_reduction_vector3_range_size : 0;
			const uint32_t scale_size = are_any_enum_flags_set(range_reduction, range_reduction_flags8::scales) ? k_clip_range_reduction_vector3_range_size : 0;
			uint32_t range_data_size = 0;

			// Only use the first segment, it contains the necessary information
			const SegmentContext& segment = clip.segments[0];
			for (const BoneStreams& bone_stream : segment.const_bone_iterator())
			{
				if (!bone_stream.is_rotation_constant)
					range_data_size += rotation_size;

				if (!bone_stream.is_translation_constant)
					range_data_size += translation_size;

				if (!bone_stream.is_scale_constant)
					range_data_size += scale_size;
			}

			return range_data_size;
		}

		inline void write_range_track_data_impl(const TrackStream& track, const TrackStreamRange& range, bool is_clip_range_data, uint8_t*& out_range_data)
		{
			const rtm::vector4f range_min = range.get_min();
			const rtm::vector4f range_extent = range.get_extent();

			if (is_clip_range_data)
			{
				const uint32_t range_member_size = sizeof(float) * 3;

				std::memcpy(out_range_data, &range_min, range_member_size);
				out_range_data += range_member_size;
				std::memcpy(out_range_data, &range_extent, range_member_size);
				out_range_data += range_member_size;
			}
			else
			{
				if (is_constant_bit_rate(track.get_bit_rate()))
				{
					const uint8_t* sample_ptr = track.get_raw_sample_ptr(0);
					std::memcpy(out_range_data, sample_ptr, sizeof(uint16_t) * 3);
					out_range_data += sizeof(uint16_t) * 3;
				}
				else
				{
					pack_vector3_u24_unsafe(range_min, out_range_data);
					out_range_data += sizeof(uint8_t) * 3;
					pack_vector3_u24_unsafe(range_extent, out_range_data);
					out_range_data += sizeof(uint8_t) * 3;
				}
			}
		}

		inline uint32_t write_range_track_data(const BoneStreams* bone_streams, const BoneRanges* bone_ranges,
			range_reduction_flags8 range_reduction, bool is_clip_range_data,
			uint8_t* range_data, uint32_t range_data_size,
			const uint32_t* output_bone_mapping, uint32_t num_output_bones)
		{
			ACL_ASSERT(range_data != nullptr, "'range_data' cannot be null!");
			(void)range_data_size;

#if defined(ACL_HAS_ASSERT_CHECKS)
			const uint8_t* range_data_end = add_offset_to_ptr<uint8_t>(range_data, range_data_size);
#endif

			const uint8_t* range_data_start = range_data;

			for (uint32_t output_index = 0; output_index < num_output_bones; ++output_index)
			{
				const uint32_t bone_index = output_bone_mapping[output_index];
				const BoneStreams& bone_stream = bone_streams[bone_index];
				const BoneRanges& bone_range = bone_ranges[bone_index];

				// normalized value is between [0.0 .. 1.0]
				// value = (normalized value * range extent) + range min
				// normalized value = (value - range min) / range extent

				if (are_any_enum_flags_set(range_reduction, range_reduction_flags8::rotations) && !bone_stream.is_rotation_constant)
				{
					const rtm::vector4f range_min = bone_range.rotation.get_min();
					const rtm::vector4f range_extent = bone_range.rotation.get_extent();

					if (is_clip_range_data)
					{
						const uint32_t range_member_size = bone_stream.rotations.get_rotation_format() == rotation_format8::quatf_full ? (sizeof(float) * 4) : (sizeof(float) * 3);

						std::memcpy(range_data, &range_min, range_member_size);
						range_data += range_member_size;
						std::memcpy(range_data, &range_extent, range_member_size);
						range_data += range_member_size;
					}
					else
					{
						if (bone_stream.rotations.get_rotation_format() == rotation_format8::quatf_full)
						{
							pack_vector4_32(range_min, true, range_data);
							range_data += sizeof(uint8_t) * 4;
							pack_vector4_32(range_extent, true, range_data);
							range_data += sizeof(uint8_t) * 4;
						}
						else
						{
							if (is_constant_bit_rate(bone_stream.rotations.get_bit_rate()))
							{
								const uint8_t* rotation = bone_stream.rotations.get_raw_sample_ptr(0);
								std::memcpy(range_data, rotation, sizeof(uint16_t) * 3);
								range_data += sizeof(uint16_t) * 3;
							}
							else
							{
								pack_vector3_u24_unsafe(range_min, range_data);
								range_data += sizeof(uint8_t) * 3;
								pack_vector3_u24_unsafe(range_extent, range_data);
								range_data += sizeof(uint8_t) * 3;
							}
						}
					}
				}

				if (are_any_enum_flags_set(range_reduction, range_reduction_flags8::translations) && !bone_stream.is_translation_constant)
					write_range_track_data_impl(bone_stream.translations, bone_range.translation, is_clip_range_data, range_data);

				if (are_any_enum_flags_set(range_reduction, range_reduction_flags8::scales) && !bone_stream.is_scale_constant)
					write_range_track_data_impl(bone_stream.scales, bone_range.scale, is_clip_range_data, range_data);

				ACL_ASSERT(range_data <= range_data_end, "Invalid range data offset. Wrote too much data.");
			}

			ACL_ASSERT(range_data == range_data_end, "Invalid range data offset. Wrote too little data.");
			return safe_static_cast<uint32_t>(range_data - range_data_start);
		}

		inline uint32_t write_clip_range_data(const clip_context& clip, range_reduction_flags8 range_reduction, uint8_t* range_data, uint32_t range_data_size, const uint32_t* output_bone_mapping, uint32_t num_output_bones)
		{
			// Only use the first segment, it contains the necessary information
			const SegmentContext& segment = clip.segments[0];

#if defined(ACL_IMPL_USE_STAGED_ANIMATED_DECOMPRESSION)
			(void)range_reduction;
			(void)range_data_size;

			// Data is ordered in groups of 4 animated sub-tracks (e.g rot0, rot1, rot2, rot3)
			// Order depends on animated track order. If we have 6 animated rotation tracks before the first animated
			// translation track, we'll have 8 animated rotation sub-tracks followed by 4 animated translation sub-tracks.
			// Once we reach the end, there is no extra padding. The last group might be less than 4 sub-tracks.
			// This is because we always process 4 animated sub-tracks at a time and cache the results.

			// Groups are written in the order of first use and as such are sorted by their lowest sub-track index.

#if defined(ACL_HAS_ASSERT_CHECKS)
			const uint8_t* range_data_end = add_offset_to_ptr<uint8_t>(range_data, range_data_size);
#endif

			const uint8_t* range_data_start = range_data;
			const rotation_format8 rotation_format = segment.bone_streams[0].rotations.get_rotation_format();	// The same for every track
			const uint32_t rotation_range_member_size = rotation_format == rotation_format8::quatf_full ? (sizeof(float) * 4) : (sizeof(float) * 3);
			const uint32_t vector3_range_member_size = sizeof(float) * 3;

			// Each range entry is a min/extent at most sizeof(vector4f) each, 32 bytes total max per sub-track, 4 sub-tracks per group
			alignas(16) uint8_t range_data_group[sizeof(rtm::vector4f) * 2 * 4];

			auto group_filter_action = [&](animation_track_type8 group_type, uint32_t bone_index)
			{
				(void)bone_index;

				if (group_type == animation_track_type8::rotation)
					return are_any_enum_flags_set(range_reduction, range_reduction_flags8::rotations);
				else if (group_type == animation_track_type8::translation)
					return are_any_enum_flags_set(range_reduction, range_reduction_flags8::translations);
				else
					return are_any_enum_flags_set(range_reduction, range_reduction_flags8::scales);
			};

			auto group_entry_action = [&](animation_track_type8 group_type, uint32_t group_size, uint32_t bone_index)
			{
				if (group_type == animation_track_type8::rotation)
				{
					const BoneRanges& bone_range = clip.ranges[bone_index];

					const rtm::vector4f range_min = bone_range.rotation.get_min();
					const rtm::vector4f range_extent = bone_range.rotation.get_extent();

					uint8_t* sub_track_range_data = &range_data_group[group_size * rotation_range_member_size * 2];
					std::memcpy(sub_track_range_data, &range_min, rotation_range_member_size);
					std::memcpy(sub_track_range_data + rotation_range_member_size, &range_extent, rotation_range_member_size);
				}
				else if (group_type == animation_track_type8::translation)
				{
					const BoneRanges& bone_range = clip.ranges[bone_index];

					const rtm::vector4f range_min = bone_range.translation.get_min();
					const rtm::vector4f range_extent = bone_range.translation.get_extent();

					uint8_t* sub_track_range_data = &range_data_group[group_size * vector3_range_member_size * 2];
					std::memcpy(sub_track_range_data, &range_min, vector3_range_member_size);
					std::memcpy(sub_track_range_data + vector3_range_member_size, &range_extent, vector3_range_member_size);
				}
				else
				{
					const BoneRanges& bone_range = clip.ranges[bone_index];

					const rtm::vector4f range_min = bone_range.scale.get_min();
					const rtm::vector4f range_extent = bone_range.scale.get_extent();

					uint8_t* sub_track_range_data = &range_data_group[group_size * vector3_range_member_size * 2];
					std::memcpy(sub_track_range_data, &range_min, vector3_range_member_size);
					std::memcpy(sub_track_range_data + vector3_range_member_size, &range_extent, vector3_range_member_size);
				}
			};

			auto group_flush_action = [&](animation_track_type8 group_type, uint32_t group_size)
			{
				const uint32_t range_member_size = group_type == animation_track_type8::rotation ? rotation_range_member_size : vector3_range_member_size;

				std::memcpy(range_data, &range_data_group[0], group_size * range_member_size * 2);
				range_data += group_size * range_member_size * 2;

				ACL_ASSERT(range_data <= range_data_end, "Invalid range data offset. Wrote too little data.");
			};

			animated_group_writer(segment, output_bone_mapping, num_output_bones, group_filter_action, group_entry_action, group_flush_action);

			ACL_ASSERT(range_data == range_data_end, "Invalid range data offset. Wrote too little data.");

			return safe_static_cast<uint32_t>(range_data - range_data_start);
#else
			return write_range_track_data(segment.bone_streams, clip.ranges, range_reduction, true, range_data, range_data_size, output_bone_mapping, num_output_bones);
#endif
		}

		inline uint32_t write_segment_range_data(const SegmentContext& segment, range_reduction_flags8 range_reduction, uint8_t* range_data, uint32_t range_data_size, const uint32_t* output_bone_mapping, uint32_t num_output_bones)
		{
#if defined(ACL_IMPL_USE_STAGED_ANIMATED_DECOMPRESSION)
			ACL_ASSERT(range_data != nullptr, "'range_data' cannot be null!");
			(void)range_reduction;
			(void)range_data_size;

			// Data is ordered in groups of 4 animated sub-tracks (e.g rot0, rot1, rot2, rot3)
			// Order depends on animated track order. If we have 6 animated rotation tracks before the first animated
			// translation track, we'll have 8 animated rotation sub-tracks followed by 4 animated translation sub-tracks.
			// Once we reach the end, there is no extra padding. The last group might be less than 4 sub-tracks.
			// This is because we always process 4 animated sub-tracks at a time and cache the results.

			// Groups are written in the order of first use and as such are sorted by their lowest sub-track index.

			// normalized value is between [0.0 .. 1.0]
			// value = (normalized value * range extent) + range min
			// normalized value = (value - range min) / range extent

#if defined(ACL_HAS_ASSERT_CHECKS)
			const uint8_t* range_data_end = add_offset_to_ptr<uint8_t>(range_data, range_data_size);
#endif

			const uint8_t* range_data_start = range_data;

			alignas(16) uint8_t range_data_group[sizeof(uint8_t) * 6 * 4];

			auto group_filter_action = [&](animation_track_type8 group_type, uint32_t bone_index)
			{
				(void)bone_index;

				if (group_type == animation_track_type8::rotation)
					return are_any_enum_flags_set(range_reduction, range_reduction_flags8::rotations);
				else if (group_type == animation_track_type8::translation)
					return are_any_enum_flags_set(range_reduction, range_reduction_flags8::translations);
				else
					return are_any_enum_flags_set(range_reduction, range_reduction_flags8::scales);
			};

			auto group_entry_action = [&](animation_track_type8 group_type, uint32_t group_size, uint32_t bone_index)
			{
				const BoneStreams& bone_stream = segment.bone_streams[bone_index];
				if (group_type == animation_track_type8::rotation)
				{
					if (is_constant_bit_rate(bone_stream.rotations.get_bit_rate()))
					{
						const uint8_t* sample = bone_stream.rotations.get_raw_sample_ptr(0);
						uint8_t* sub_track_range_data = &range_data_group[group_size * sizeof(uint8_t) * 6];
						std::memcpy(sub_track_range_data, sample, sizeof(uint8_t) * 6);
					}
					else
					{
						const BoneRanges& bone_range = segment.ranges[bone_index];

						const rtm::vector4f range_min = bone_range.rotation.get_min();
						const rtm::vector4f range_extent = bone_range.rotation.get_extent();

						uint8_t* sub_track_range_data = &range_data_group[group_size * sizeof(uint8_t) * 6];
						pack_vector3_u24_unsafe(range_min, sub_track_range_data);
						pack_vector3_u24_unsafe(range_extent, sub_track_range_data + sizeof(uint8_t) * 3);
					}
				}
				else if (group_type == animation_track_type8::translation)
				{
					if (is_constant_bit_rate(bone_stream.translations.get_bit_rate()))
					{
						const uint8_t* sample = bone_stream.translations.get_raw_sample_ptr(0);
						uint8_t* sub_track_range_data = &range_data_group[group_size * sizeof(uint8_t) * 6];
						std::memcpy(sub_track_range_data, sample, sizeof(uint8_t) * 6);
					}
					else
					{
						const BoneRanges& bone_range = segment.ranges[bone_index];

						const rtm::vector4f range_min = bone_range.translation.get_min();
						const rtm::vector4f range_extent = bone_range.translation.get_extent();

						uint8_t* sub_track_range_data = &range_data_group[group_size * sizeof(uint8_t) * 6];
						pack_vector3_u24_unsafe(range_min, sub_track_range_data);
						pack_vector3_u24_unsafe(range_extent, sub_track_range_data + sizeof(uint8_t) * 3);
					}
				}
				else
				{
					if (is_constant_bit_rate(bone_stream.scales.get_bit_rate()))
					{
						const uint8_t* sample = bone_stream.scales.get_raw_sample_ptr(0);
						uint8_t* sub_track_range_data = &range_data_group[group_size * sizeof(uint8_t) * 6];
						std::memcpy(sub_track_range_data, sample, sizeof(uint8_t) * 6);
					}
					else
					{
						const BoneRanges& bone_range = segment.ranges[bone_index];

						const rtm::vector4f range_min = bone_range.scale.get_min();
						const rtm::vector4f range_extent = bone_range.scale.get_extent();

						uint8_t* sub_track_range_data = &range_data_group[group_size * sizeof(uint8_t) * 6];
						pack_vector3_u24_unsafe(range_min, sub_track_range_data);
						pack_vector3_u24_unsafe(range_extent, sub_track_range_data + sizeof(uint8_t) * 3);
					}
				}
			};

			auto group_flush_action = [&](animation_track_type8 group_type, uint32_t group_size)
			{
				(void)group_type;

				std::memcpy(range_data, &range_data_group[0], group_size * sizeof(uint8_t) * 6);
				range_data += group_size * sizeof(uint8_t) * 6;

				ACL_ASSERT(range_data <= range_data_end, "Invalid range data offset. Wrote too little data.");
			};

			animated_group_writer(segment, output_bone_mapping, num_output_bones, group_filter_action, group_entry_action, group_flush_action);

			ACL_ASSERT(range_data == range_data_end, "Invalid range data offset. Wrote too little data.");

			return safe_static_cast<uint32_t>(range_data - range_data_start);
#else
			return write_range_track_data(segment.bone_streams, segment.ranges, range_reduction, false, range_data, range_data_size, output_bone_mapping, num_output_bones);
#endif
		}
	}
}

ACL_IMPL_FILE_PRAGMA_POP
