#ifndef _TUTTLE_COMMON_SYSTEM_COMPATIBILITY_HPP_
#define _TUTTLE_COMMON_SYSTEM_COMPATIBILITY_HPP_

#include <cstddef>

#ifdef _MSC_VER
#include <BaseTsd.h>
#else
#include <unistd.h>
#endif

// compatibility problems...
namespace std {
#ifdef _MSC_VER
	typedef SSIZE_T ssize_t;
#else
	typedef ::ssize_t ssize_t;
#endif
}


#endif

