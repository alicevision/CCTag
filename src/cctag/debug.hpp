#ifndef _ROM_DEBUG_HPP_
#define _ROM_DEBUG_HPP_

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

/// @see ROM_COUT
#define ROM_COUT_DEBUG ROM_COUT

/// @see  ROM_COUT_INFOS
#define ROM_COUT_INFOS_DEBUG ROM_COUT_INFOS

/// @see  ROM_COUT_INFOS
#define ROM_COUT_WITHINFOS_DEBUG ROM_COUT_WITHINFOS

/// @see ROM_IF_DEBUG
#define ROM_IF_DEBUG(... )  __VA_ARGS__

#define ROM_COUT_X_DEBUG ROM_COUT_X
#define ROM_COUT_VAR_DEBUG ROM_COUT_VAR
#define ROM_COUT_VAR2_DEBUG ROM_COUT_VAR2
#define ROM_COUT_VAR3_DEBUG ROM_COUT_VAR3
#define ROM_COUT_VAR4_DEBUG ROM_COUT_VAR4
#define ROM_COUT_INFOS_DEBUG ROM_COUT_INFOS
#define ROM_COUT_WITHINFOS_DEBUG ROM_COUT_WITHINFOS
#define ROM_COUT_WARNING_DEBUG ROM_COUT_WARNING
#define ROM_COUT_ERROR_DEBUG ROM_COUT_ERROR
#define ROM_COUT_FATALERROR_DEBUG ROM_COUT_FATALERROR
#define ROM_COUT_EXCEPTION_DEBUG ROM_COUT_EXCEPTION

//#ifdef __WINDOWS__
// #include "windows/MemoryLeaks.hpp"
//#endif

#endif

