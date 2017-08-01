/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_EXCEPTIONS_HPP_
#define _CCTAG_EXCEPTIONS_HPP_

#include "Backtrace.hpp"

#include <boost/exception/diagnostic_information.hpp>
#include <boost/exception/errinfo_file_name.hpp>
#include <boost/exception/exception.hpp>
#include <boost/exception/exception.hpp>
#include <boost/exception/get_error_info.hpp>
#include <boost/exception/info.hpp>
#include <boost/exception/info.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/version.hpp>
#include <boost/version.hpp>

#include <exception>
#include <stdexcept>
#include <string>

#if( BOOST_VERSION >= 105400 )
#include <boost/exception/to_string_stub.hpp>
#endif

#include <sstream>
#include <string>

#ifndef SWIG
namespace boost {

struct error_info_sstream
{
	using value_type = std::ostringstream;
	value_type _v;
};

inline std::string to_string( const error_info_sstream& x )
{
	return x._v.str();
}

inline std::ostream& operator<<( std::ostream& os, const error_info_sstream& x )
{
	os << x._v.str();
	return os;
}

template<class Tag>
class error_info<Tag, error_info_sstream>: public exception_detail::error_info_base
{
public:
	using T = boost::error_info_sstream;
	using This = error_info<Tag, T>;
	using value_type = T;

	error_info() {}
	error_info( const This& v )
	{
		_value._v << v._value._v.str();
	}

	template<typename V>
	error_info( const V& value )
	{
		_value._v << value;
	}

	~error_info() throw( ) {}

	template<typename V>
	This& operator+( const V& v )
	{
		_value._v << v;
		return *this;
	}

	const value_type& value() const { return _value; }
	value_type&       value()       { return _value; }

private:
	
	#if( BOOST_VERSION >= 105400 )
	inline std::string name_value_string() const
	{
		return to_string_stub(*this);
	}
	#elif( BOOST_VERSION >= 104300 )
	std::string tag_typeid_name() const { return tag_type_name<Tag>(); }
	#else
	char const* tag_typeid_name() const { return tag_type_name<Tag>(); }
	#endif
	std::string value_as_string() const { return _value._v.str(); }

	value_type _value;
};
}
#endif

#ifdef CCTAG_NO_EXCEPTIONS
#define CCTAG_NO_TRY_CATCH
#define CCTAG_THROW(...)
#else
#define CCTAG_THROW BOOST_THROW_EXCEPTION
#endif

#define CCTAG_FORCE_COUT_BOOST_EXCEPTION( e )  \
    ::std::cerr << "Exception:" << \
    ::std::endl << CCTAG_INFOS << \
    ::std::endl << "\t" << ::boost::diagnostic_information( e )

#define CCTAG_FORCE_COUT_CURRENT_EXCEPTION  \
    ::std::cerr << "Exception:" << \
    ::std::endl << CCTAG_INFOS << \
    ::std::endl << "\t" << ::boost::current_exception_diagnostic_information()

#ifndef CCTAG_NO_TRY_CATCH
#	define CCTAG_TRY try
#	define CCTAG_CATCH(x) catch( x )
#	define CCTAG_RETHROW throw

#define CCTAG_COUT_BOOST_EXCEPTION(e) CCTAG_FORCE_COUT_BOOST_EXCEPTION(e)
#define CCTAG_COUT_CURRENT_EXCEPTION CCTAG_FORCE_COUT_CURRENT_EXCEPTION

#else
#    if BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x564))
#        define CCTAG_TRY if( "" )
#        define CCTAG_CATCH(x) else if( !"" )
#    else
#        define CCTAG_TRY if( true )
#        define CCTAG_CATCH(x) else if( false )
#    endif
#	define CCTAG_RETHROW
#define CCTAG_COUT_BOOST_EXCEPTION(e)
#define CCTAG_COUT_CURRENT_EXCEPTION
#endif





namespace cctag {

/**
 * @brief To add quotes around a string.
 * @example quotes("toto") -> "\"toto\""
 */
inline std::string quotes( const std::string& s )
{
	return "\"" + s + "\"";
}

namespace exception {

#ifndef SWIG
/**
 * @brief Like a stringstream but using "operator+" instead of "operator<<".
 * Use a stringstream internally.
 */

/**
 * @brief Standard tags you can use to add informations to your exceptions.
 *
 * @remark We use lower camel case for tags,
 *	to keep a difference between tags and exceptions.
 */
/// @{
/**
 * @brief If you catch an error at the top level, you can print this information to the user.
 * @remark User information.
 */
typedef ::boost::error_info<struct tag_userMessage, ::boost::error_info_sstream> user;

/**
 * @brief This is detailed informations for developpers.
 * Not always a real human readable message :)
 * @remark Dev information.
 */
//typedef ::boost::error_info<struct tag_message,std::string> dev;
typedef ::boost::error_info<struct tag_devMessage, ::boost::error_info_sstream> dev;
//typedef ::boost::error_info_sstream<struct tag_message> dev;

/**
 * @brief The algorithm
 * @remark Dev information.
 */
typedef ::boost::error_info<struct tag_algorithm, ::std::string> algorithm;
/**
 * @brief Time.
 * @remark Dev or user information.
 */
typedef ::boost::error_info<struct tag_time, float> time;
/**
 * @brief Frame number.
 * @remark Dev or user information.
 */
typedef ::boost::error_info<struct tag_frame, long int> frame;
/**
 * @brief Problem with a file.
 * @remark User information.
 */
using filename = ::boost::errinfo_file_name;
/// @}
#endif

/** @brief Common exception, base of all exceptions inside the project */
struct Common
	: virtual public ::std::exception
	, virtual public ::boost::exception
	, virtual public ::boost::backtrace
{};

/// @brief All typed exceptions
/// @{

/** @brief Status error code for a failed operation */
struct Failed : virtual public Common {};

/**
 * @brief Status error code for a fatal error
 *
 * Only returned in the case where we cannot continue to function and needs to be restarted.
 */
struct Fatal : virtual public Common {};

/** @brief Status error code for an operation on or request for an unknown object */
struct Unknown : virtual public Common {};

/** @brief Status error code for an unsupported feature/operation */
struct Unsupported : virtual public Common {};

/** @brief Status error code for an operation attempting to create something that exists */
struct Exists : virtual public Common {};

/** @brief Status error code indicating that something failed due to memory shortage */
struct Memory : virtual public Common {};

/** @brief Status error code for an operation on a bad handle */
struct BadHandle : virtual public Common {};

struct NullPtr : virtual public BadHandle {};

/** @brief Status error code for an operation on a bad handle */
struct BadAlloc : virtual public Common {};

/** @brief Status error code indicating that a given index was invalid or unavailable */
struct BadIndex : virtual public Common {};

struct OutOfRange : virtual public BadIndex {};

/** @brief Status error code indicating that something failed due an illegal value */
struct Value : virtual public Common {};

struct Argument : virtual public Value {};

struct UnsupportedFormat : virtual public Common {};

/** @brief Error code for incorrect image formats */
struct ImageFormat : virtual public UnsupportedFormat {};

struct UnmatchedSizes : virtual public Unsupported {};

struct BadSize : virtual public Unsupported {};

struct DivideZero : virtual public Unsupported {};

/// @}

/// @brief Other exceptions
/// @{

/**
 * @brief The class serves as the base class for all exceptions thrown to report errors presumably detectable before the program executes, such as violations of logical preconditions (cf. std::logic_error).
 * @remark With this exception, you normally have a "user" tag message.
 */
struct Logic : virtual public Value {};

/**
 * @brief Something that should never appends.
 *        These exceptions may be replaced by assertions,
 *        but we prefer to keep a runtime check even in release (for the moment).
 * @remark With this exception, you should have a "dev" tag message.
 */
struct Bug : virtual public Value {};

struct NotImplemented : virtual public Unsupported {};

/** @brief Unknown error inside a conversion. */
struct BadConversion : virtual public Value {};

/**
 * @brief File manipulation error.
 * eg. read only, file doesn't exists, etc.
 */
struct File : virtual public Value
{
	File()
	{}
	File( const std::string& path )
	{
		*this << filename(path);
	}
};

/**
 * @brief File doesn't exists.
 */
struct FileNotExist : virtual public File
{
	FileNotExist()
	{}
	FileNotExist( const std::string& path )
	: File( path )
	{}
};

/**
 * @brief Directory doesn't exists.
 */
struct NoDirectory : virtual public File
{
	NoDirectory()
	{}
	NoDirectory( const std::string& path )
	: File( path )
	{}
};

/**
 * @brief Read only file.
 */
struct ReadOnlyFile : virtual public File
{
	ReadOnlyFile()
	{}
	ReadOnlyFile( const std::string& path )
	: File( path )
	{}
};
/// @}

}
}

#endif
