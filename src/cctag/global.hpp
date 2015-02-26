#ifndef _CCTAG_GLOBAL_HPP_
#define _CCTAG_GLOBAL_HPP_

//#define ROM_NO_COUT

#include "progBase/system/system.hpp"

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
#    define ROM_FORCEINLINE __forceinline
#  elif defined( __GNUC__ ) && __GNUC__ > 3
#    define ROM_FORCEINLINE inline __attribute__ ( ( always_inline ) )
#  else
#    define ROM_FORCEINLINE inline
#  endif
#else
#  define ROM_FORCEINLINE inline
#endif

#ifdef DEBUG
 #include "debug.hpp"
#else
 #include "release.hpp"
#endif

#ifndef ROM_PAUSE
#ifdef DEBUG
#define ROM_PAUSE  ::std::cin.ignore().get();
#else
#define ROM_PAUSE
#endif
#endif

#ifndef ROM_COUT

/**
 * @def   ROM_INFOS
 * @brief informations : filename, line number, function name
 **/
 #define ROM_INFOS  "file: " << __FILE__ << ",  line: " << __LINE__ << ::std::endl << "function: " << BOOST_CURRENT_FUNCTION

 #define ROM_VAR( a )  # a << ": " << a
 #define ROM_VAR2( a, b )  # a << ": " << a << ", " << # b << ": " << b
 #define ROM_VAR3( a, b, c )  # a << ": " << a << ", " << # b << ": " << b << ", " << # c << ": " << c
 #define ROM_VAR4( a, b, c, d )  # a << ": " << a << ", " << # b << ": " << b << ", " << # c << ": " << c << ", " << # d << ": " << d
 #define ROM_VAR_ENDL( a )  # a << ":" << ::std::endl << a

#ifndef ROM_NO_COUT
/**
 * @param[in] ... : all parameters with an operator << defined
 * @brief terminal display
 **/
 #define ROM_COUT(... )  ::std::cout << __VA_ARGS__ << ::std::endl
 #define ROM_COUT_NOENDL(... )  ::std::cout << __VA_ARGS__
 #define ROM_CERR(... )  ::std::cerr << __VA_ARGS__ << ::std::endl
 #define ROM_CERR_NOENDL(... )  ::std::cerr << __VA_ARGS__

 #define ROM_COUT_X( N, ... ) \
    for( unsigned int i = 0; i < N; ++i ) { ::std::cout << __VA_ARGS__; } \
    ::std::cout << ::std::endl

 #define ROM_CERR_X( N, ... ) \
    for( unsigned int i = 0; i < N; ++i ) { ::std::cerr << __VA_ARGS__; } \
    ::std::cerr << ::std::endl

#else
 #define ROM_COUT(...)
 #define ROM_COUT_NOENDL(... )
 #define ROM_CERR(...)
 #define ROM_COUT_X( N, ... )
#endif


 #define ROM_COUT_VAR( a )  ROM_COUT( ROM_VAR( a ) )
 #define ROM_COUT_VAR2( a, b )  ROM_COUT( ROM_VAR2( a, b ) )
 #define ROM_COUT_VAR3( a, b, c )  ROM_COUT( ROM_VAR3( a, b, c ) )
 #define ROM_COUT_VAR4( a, b, c, d )  ROM_COUT( ROM_VAR4( a, b, c, d ) )

/**
 * @brief terminal information display
 **/
 #define ROM_COUT_INFOS ROM_COUT( ROM_INFOS )

/**
 * @param[in] ... : all parameters with an operator << defined
 * @brief terminal information display
 **/
 #define ROM_COUT_WITHINFOS(... )  \
    ROM_COUT( ROM_INFOS << \
          ::std::endl << "\t" << __VA_ARGS__ )

 #define ROM_COUT_WARNING(... )  \
    ROM_CERR( "Warning:" << \
    ::std::endl << ROM_INFOS << \
    ::std::endl << "\t" << __VA_ARGS__  )

 #define ROM_COUT_ERROR(... )  \
    ROM_CERR( "Error:" << \
    ::std::endl << ROM_INFOS << \
    ::std::endl << "\t" << __VA_ARGS__  )

 #define ROM_COUT_FATALERROR(... )  \
    ROM_CERR( "Fatal error:" << \
    ::std::endl << ROM_INFOS << \
    ::std::endl << "\t" << __VA_ARGS__  )

#endif

////////////////////////////////////////////////////////////////////////////////
// Some specifics things to debug or release version
#ifdef DEBUG
 #include "debug.hpp"
#else
 #include "release.hpp"
#endif

////////////////////////////////////////////////////////////////////////////////
// ROM_TCOUT* defines are used by developpers for temporary displays during development stages.
// They are removed in production mode.
#ifndef ROM_PRODUCTION
	#define ROM_TCOUT ROM_COUT
	#define ROM_TCOUT_NOENDL ROM_COUT_NOENDL
	#define ROM_TCOUT_X ROM_COUT_X
	#define ROM_TCOUT_VAR ROM_COUT_VAR
	#define ROM_TCOUT_VAR2 ROM_COUT_VAR2
	#define ROM_TCOUT_VAR3 ROM_COUT_VAR3
	#define ROM_TCOUT_VAR4 ROM_COUT_VAR4
	#define ROM_TCOUT_INFOS ROM_COUT_INFOS
	#define ROM_TCOUT_WITHINFOS ROM_COUT_WITHINFOS
#else
	#define ROM_TCOUT ROM_COUT_DEBUG
	#define ROM_TCOUT_X ROM_COUT_X_DEBUG
	#define ROM_TCOUT_VAR ROM_COUT_VAR_DEBUG
	#define ROM_TCOUT_VAR2 ROM_COUT_VAR2_DEBUG
	#define ROM_TCOUT_VAR3 ROM_COUT_VAR3_DEBUG
	#define ROM_TCOUT_VAR4 ROM_COUT_VAR4_DEBUG
	#define ROM_TCOUT_INFOS ROM_COUT_INFOS_DEBUG
	#define ROM_TCOUT_WITHINFOS ROM_COUT_WITHINFOS_DEBUG
#endif



#ifdef USER_LILIAN
	#define ROM_COUT_LILIAN  ROM_COUT
#else
	#define ROM_COUT_LILIAN( ... )
#endif

#endif
