/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

// a macro that switches on printing in cctag
#define CCTAG_FIND_RANDOMNESS

#undef DEBUG_WRITE_ORIGINAL_AS_PGM
#undef DEBUG_WRITE_ORIGINAL_AS_ASCII
#undef DEBUG_WRITE_DX_AS_PGM
#undef DEBUG_WRITE_DX_AS_ASCII
#undef DEBUG_WRITE_DY_AS_PGM
#undef DEBUG_WRITE_DY_AS_ASCII
#undef DEBUG_WRITE_MAG_AS_PGM
#undef DEBUG_WRITE_MAG_AS_ASCII
#undef DEBUG_WRITE_MAP_AS_PGM
#undef DEBUG_WRITE_MAP_AS_ASCII
#undef DEBUG_WRITE_HYSTEDGES_AS_PGM
#undef DEBUG_WRITE_EDGES_AS_PGM
#undef DEBUG_WRITE_EDGELIST_AS_PPM
#undef DEBUG_WRITE_EDGELIST_AS_ASCII
#undef DEBUG_WRITE_VOTERS_AS_PPM
#undef DEBUG_WRITE_VOTERS_AS_ASCII
#define DEBUG_WRITE_CHOSEN_AS_PPM
#define DEBUG_WRITE_CHOSEN_VOTERS_AS_ASCII
#define DEBUG_WRITE_CHOSEN_ELECTED_AS_ASCII
// WARNING: follow 4 show bug in separable compilation
#define DEBUG_WRITE_LINKED_AS_PPM
#define DEBUG_WRITE_LINKED_AS_PPM_INTENSE
#define DEBUG_WRITE_LINKED_AS_ASCII
#define DEBUG_WRITE_LINKED_AS_ASCII_INTENSE

//#define DEBUG_LINKED_USE_INT4_BUFFER

/* Separable compilation allows one kernel to instantiate
 * others. That avoids complexity on the host side when,
 * e.g., GPU-side counters need to be checked before starting
 * a new kernel.
 *
 * Enable only when you know what you are doing.
 * The separable compilation can only work if you compile for a
 * CUDA compute capability of 3.5 or above.
 * The benefits of separable compilation are not so obvious that
 * we have written a runtime check for it.
 */
// #undef USE_SEPARABLE_COMPILATION
#undef USE_SEPARABLE_COMPILATION_FOR_HYST
#undef USE_SEPARABLE_COMPILATION_FOR_GRADDESC
#undef USE_SEPARABLE_COMPILATION_FOR_VOTE_LINE
#undef USE_SEPARABLE_COMPILATION_FOR_EVAL

/* Affects tag.cu.
.* A Frame used two CUDA streams, one for upload and kernels, another
 * one for downloads.
 * The download stream can be shared between Frame data structures. It is
 * uncertain if that has any negative side-effects, because the PCIe
 * bus is congested when they compete.
 */
#undef USE_ONE_DOWNLOAD_STREAM

/* Affects frame_export.cu.
 * We must sort all potential inner points, but must we sort the
 * edge point coordinates to avoid randomness? If it appears that
 * this is the case, #define this.
 */
//#define SORT_ALL_EDGECOORDS_IN_EXPORT

/* Space for nearby points must be allocated in pinned memory.
 * The number of such objects must be limited, and this is the
 * limits.
 */
#define MAX_MARKER_FOR_IDENT 60

/* How many parallel pipelines can we have?
 */
#define MAX_PIPES	4
