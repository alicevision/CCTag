#pragma once

#define DEBUG_WRITE_ORIGINAL_AS_PGM
#define DEBUG_WRITE_ORIGINAL_AS_ASCII
#define DEBUG_WRITE_DX_AS_PGM
#define DEBUG_WRITE_DX_AS_ASCII
#define DEBUG_WRITE_DY_AS_PGM
#define DEBUG_WRITE_DY_AS_ASCII
#define DEBUG_WRITE_MAG_AS_PGM
#define DEBUG_WRITE_MAG_AS_ASCII
#define DEBUG_WRITE_MAP_AS_PGM
#define DEBUG_WRITE_MAP_AS_ASCII
#define DEBUG_WRITE_HYSTEDGES_AS_PGM
#define DEBUG_WRITE_EDGES_AS_PGM
#define DEBUG_WRITE_EDGELIST_AS_PPM
#define DEBUG_WRITE_EDGELIST_AS_ASCII
#define DEBUG_WRITE_VOTERS_AS_PPM
#define DEBUG_WRITE_CHOSEN_AS_PPM
#define DEBUG_WRITE_CHOSEN_VOTERS_AS_ASCII
#define DEBUG_WRITE_CHOSEN_ELECTED_AS_ASCII
#define DEBUG_WRITE_LINKED_AS_PPM
#define DEBUG_WRITE_LINKED_AS_PPM_INTENSE
#define DEBUG_WRITE_LINKED_AS_ASCII
#define DEBUG_WRITE_LINKED_AS_ASCII_INTENSE

#define DEBUG_LINKED_USE_INT4_BUFFER

/* Separable compilation allows one kernel to instantiate
 * others. That avoids complexity on the host side when,
 * e.g., GPU-side counters need to be checked before starting
 * a new kernel.
 */
#undef USE_SEPARABLE_COMPILATION
#undef USE_SEPARABLE_COMPILATION_IN_GRADDESC

/* Init _d_intermediate to 0 before uploading. Wastes time,
 * for debugging only. Required because of crash -O3 but not
 * with -G
 */
#define DEBUG_FRAME_UPLOAD_CUTS

/* Define if you want to compute identity both on GPU and CPU.
 * The GPU version takes precedence.
 * Otherwise, GPU is used if cudaPipe exists (the alternative,
 * param.useCuda == false is broken in the optim_identify_gpu
 * branch).
 */
#undef CPU_GPU_COST_FUNCTION_COMPARE

