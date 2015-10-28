#ifndef _CCTAG_GLOBAL_HPP_
#define _CCTAG_GLOBAL_HPP_

#undef CCTAG_NO_COUT

#include <cctag/progBase/system/system.hpp>

////////////////////////////////////////////////////////////////////////////////
// Assert needs to be everywhere
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>
#include <boost/current_function.hpp>

////////////////////////////////////////////////////////////////////////////////
// compatibility problems...
#include <cstddef>
namespace std {
#ifdef _MSC_VER
	typedef SSIZE_T ssize_t;
#else
	//#ifdef __ssize_t_defined
	//typedef __ssize_t ssize_t;
	//#else
	using ::ssize_t;
	//#endif
#endif
}


////////////////////////////////////////////////////////////////////////////////
// Define functions to display infos in the console
#include <iostream>

#ifdef NDEBUG
#  if defined( _MSC_VER )
#    define CCTAG_FORCEINLINE __forceinline
#  elif defined( __GNUC__ ) && __GNUC__ > 3
#    define CCTAG_FORCEINLINE inline __attribute__ ( ( always_inline ) )
#  else
#    define CCTAG_FORCEINLINE inline
#  endif
#else
#  define CCTAG_FORCEINLINE inline
#endif

#ifdef DEBUG
 #include "debug.hpp"
#else
 #include "release.hpp"
#endif

#ifndef CCTAG_PAUSE
#ifdef DEBUG
#define CCTAG_PAUSE  ::std::cin.ignore().get();
#else
#define CCTAG_PAUSE
#endif
#endif

#ifndef CCTAG_COUT

/**
 * @def   CCTAG_INFOS
 * @brief informations : filename, line number, function name
 **/
 #define CCTAG_INFOS  "file: " << __FILE__ << ",  line: " << __LINE__ << ::std::endl << "function: " << BOOST_CURRENT_FUNCTION

 #define CCTAG_VAR( a )  # a << ": " << a
 #define CCTAG_VAR2( a, b )  # a << ": " << a << ", " << # b << ": " << b
 #define CCTAG_VAR3( a, b, c )  # a << ": " << a << ", " << # b << ": " << b << ", " << # c << ": " << c
 #define CCTAG_VAR4( a, b, c, d )  # a << ": " << a << ", " << # b << ": " << b << ", " << # c << ": " << c << ", " << # d << ": " << d
 #define CCTAG_VAR_ENDL( a )  # a << ":" << ::std::endl << a

#ifndef CCTAG_NO_COUT
/**
 * @param[in] ... : all parameters with an operator << defined
 * @brief terminal display
 **/
 #define CCTAG_COUT(... )  ::std::cout << __FILE__ << ":" << __LINE__ << ": " << __VA_ARGS__ << ::std::endl
 #define CCTAG_COUT_NOENDL(... )  ::std::cout << __VA_ARGS__
 #define CCTAG_CERR(... )  ::std::cerr << __VA_ARGS__ << ::std::endl
 #define CCTAG_CERR_NOENDL(... )  ::std::cerr << __VA_ARGS__

 #define CCTAG_COUT_X( N, ... ) \
    for( unsigned int i = 0; i < N; ++i ) { ::std::cout << __VA_ARGS__; } \
    ::std::cout << ::std::endl

 #define CCTAG_CERR_X( N, ... ) \
    for( unsigned int i = 0; i < N; ++i ) { ::std::cerr << __VA_ARGS__; } \
    ::std::cerr << ::std::endl

#else
 #define CCTAG_COUT(...)
 #define CCTAG_COUT_NOENDL(... )
 #define CCTAG_CERR(...)
 #define CCTAG_COUT_X( N, ... )
#endif


 #define CCTAG_COUT_VAR( a )  CCTAG_COUT( CCTAG_VAR( a ) )
 #define CCTAG_COUT_VAR2( a, b )  CCTAG_COUT( CCTAG_VAR2( a, b ) )
 #define CCTAG_COUT_VAR3( a, b, c )  CCTAG_COUT( CCTAG_VAR3( a, b, c ) )
 #define CCTAG_COUT_VAR4( a, b, c, d )  CCTAG_COUT( CCTAG_VAR4( a, b, c, d ) )

/**
 * @brief terminal information display
 **/
 #define CCTAG_COUT_INFOS CCTAG_COUT( CCTAG_INFOS )

/**
 * @param[in] ... : all parameters with an operator << defined
 * @brief terminal information display
 **/
 #define CCTAG_COUT_WITHINFOS(... )  \
    CCTAG_COUT( CCTAG_INFOS << \
          ::std::endl << "\t" << __VA_ARGS__ )

 #define CCTAG_COUT_WARNING(... )  \
    CCTAG_CERR( "Warning:" << \
    ::std::endl << CCTAG_INFOS << \
    ::std::endl << "\t" << __VA_ARGS__  )

 #define CCTAG_COUT_ERROR(... )  \
    CCTAG_CERR( "Error:" << \
    ::std::endl << CCTAG_INFOS << \
    ::std::endl << "\t" << __VA_ARGS__  )

 #define CCTAG_COUT_FATALERROR(... )  \
    CCTAG_CERR( "Fatal error:" << \
    ::std::endl << CCTAG_INFOS << \
    ::std::endl << "\t" << __VA_ARGS__  )

#endif

////////////////////////////////////////////////////////////////////////////////
// Some specifics things to debug or release version
#ifdef DEBUG
 #include "debug.hpp"
#else
 #include "release.hpp"
#endif

#ifdef CCTAG_OPTIM
 #define CCTAG_COUT_OPTIM CCTAG_COUT
 #define CCTAG_COUT_VAR_OPTIM CCTAG_COUT_VAR
#else
 #define CCTAG_COUT_OPTIM(...)
 #define CCTAG_COUT_VAR_OPTIM(...)
#endif

////////////////////////////////////////////////////////////////////////////////
// CCTAG_TCOUT* defines are used by developpers for temporary displays during development stages.
// They are removed in production mode.
#ifndef CCTAG_PRODUCTION
	#define CCTAG_TCOUT CCTAG_COUT
	#define CCTAG_TCOUT_NOENDL CCTAG_COUT_NOENDL
	#define CCTAG_TCOUT_X CCTAG_COUT_X
	#define CCTAG_TCOUT_VAR CCTAG_COUT_VAR
	#define CCTAG_TCOUT_VAR2 CCTAG_COUT_VAR2
	#define CCTAG_TCOUT_VAR3 CCTAG_COUT_VAR3
	#define CCTAG_TCOUT_VAR4 CCTAG_COUT_VAR4
	#define CCTAG_TCOUT_INFOS CCTAG_COUT_INFOS
	#define CCTAG_TCOUT_WITHINFOS CCTAG_COUT_WITHINFOS
#else
	#define CCTAG_TCOUT CCTAG_COUT_DEBUG
	#define CCTAG_TCOUT_X CCTAG_COUT_X_DEBUG
	#define CCTAG_TCOUT_VAR CCTAG_COUT_VAR_DEBUG
	#define CCTAG_TCOUT_VAR2 CCTAG_COUT_VAR2_DEBUG
	#define CCTAG_TCOUT_VAR3 CCTAG_COUT_VAR3_DEBUG
	#define CCTAG_TCOUT_VAR4 CCTAG_COUT_VAR4_DEBUG
	#define CCTAG_TCOUT_INFOS CCTAG_COUT_INFOS_DEBUG
	#define CCTAG_TCOUT_WITHINFOS CCTAG_COUT_WITHINFOS_DEBUG
#endif



#ifdef USER_LILIAN
	#define CCTAG_COUT_LILIAN  CCTAG_COUT
#else
	#define CCTAG_COUT_LILIAN( ... )
#endif

#endif
