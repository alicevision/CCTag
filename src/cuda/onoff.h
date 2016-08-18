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

/* Although some GPU code exists, it is too slow and edge
 * linking is still done on the host side.
 */
#define EDGE_LINKING_HOST_SIDE

//#define DEBUG_LINKED_USE_INT4_BUFFER

/* Separable compilation allows one kernel to instantiate
 * others. That avoids complexity on the host side when,
 * e.g., GPU-side counters need to be checked before starting
 * a new kernel.
 */
// #undef USE_SEPARABLE_COMPILATION
#define USE_SEPARABLE_COMPILATION_FOR_HYST
#define USE_SEPARABLE_COMPILATION_FOR_GRADDESC
#define USE_SEPARABLE_COMPILATION_FOR_VOTE_LINE
#define USE_SEPARABLE_COMPILATION_FOR_SORT_UNIQ
#define USE_SEPARABLE_COMPILATION_FOR_EVAL
#define USE_SEPARABLE_COMPILATION_FOR_VOTE_IF


/* CUB functions always take a last parameters true or false.
 * If it is true, they run synchronously and print some debug
 * info.
 */
#define DEBUG_CUB_FUNCTIONS false

/* CUB RadixSort requires the DoubleBuffer structure and annoying
 * host-side sync in the CUB version that comes with CUDA 7.0.
 * CUDA 7.5 or standalone CUB 1.4.1 allow an output buffer.
 * However, so far they don't work.
 */
#undef RADIX_WITHOUT_DOUBLEBUFFER

/* For the CUB version included with CUDA 7.0, it was possible to
 * pass an arbitrary device pointer pointing to sufficiently large
 * memory.
 * Standalone CUB 1.4.1 fails unless the init call for determining
 * intermediate buffer size is made.
 */
#define CUB_INIT_CALLS

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

