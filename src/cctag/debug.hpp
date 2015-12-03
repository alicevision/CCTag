#ifndef _CCTAG_DEBUG_HPP_
#define _CCTAG_DEBUG_HPP_

// pre-processeur directives :
//	# : convert to string
//	## : concatenate (or before __VA_ARGS__ to indicate that it may be empty)
//	__FILE__ : filename
//	__LINE__ : line number
//	__FUNCTION__ : function declaration
//	__PRETTY_FUNCTION__ : function name
//	__DATE__ : "Mmm dd yyyy"
//	__TIME__ : "hh:mm:ss"

//_____________________________________________________________________________
// Macros to output on terminal only in debug mode

/// @see CCTAG_COUT
#define CCTAG_COUT_DEBUG CCTAG_COUT

/// @see  CCTAG_COUT_INFOS
#define CCTAG_COUT_INFOS_DEBUG CCTAG_COUT_INFOS

/// @see  CCTAG_COUT_INFOS
#define CCTAG_COUT_WITHINFOS_DEBUG CCTAG_COUT_WITHINFOS

/// @see CCTAG_IF_DEBUG
#define CCTAG_IF_DEBUG(... )  __VA_ARGS__

#define CCTAG_COUT_X_DEBUG CCTAG_COUT_X
#define CCTAG_COUT_VAR_DEBUG CCTAG_COUT_VAR
#define CCTAG_COUT_VAR2_DEBUG CCTAG_COUT_VAR2
#define CCTAG_COUT_VAR3_DEBUG CCTAG_COUT_VAR3
#define CCTAG_COUT_VAR4_DEBUG CCTAG_COUT_VAR4
#define CCTAG_COUT_INFOS_DEBUG CCTAG_COUT_INFOS
#define CCTAG_COUT_WITHINFOS_DEBUG CCTAG_COUT_WITHINFOS
#define CCTAG_COUT_WARNING_DEBUG CCTAG_COUT_WARNING
#define CCTAG_COUT_ERROR_DEBUG CCTAG_COUT_ERROR
#define CCTAG_COUT_FATALERROR_DEBUG CCTAG_COUT_FATALERROR
#define CCTAG_COUT_EXCEPTION_DEBUG CCTAG_COUT_EXCEPTION

#ifndef WITH_CUDA
#define POP_INFO(s)  std::cerr << __FILE__ << ":" << __LINE__ << " INFO: " << s << std::endl
#else
#define POP_INFO(s)
#endif
#endif // POP_INFO
#define POP_ERROR std::cerr << __FILE__ << ":" << __LINE__ << " ERROR: "
#define POP_ENTER std::cerr << __FILE__ << ":" << __LINE__ << " entering " << __func__ << std::endl
#define POP_LEAVE std::cerr << __FILE__ << ":" << __LINE__ << " leaving " << __func__ << std::endl

//#ifdef __WINDOWS__
// #include "windows/MemoryLeaks.hpp"
//#endif

#endif

