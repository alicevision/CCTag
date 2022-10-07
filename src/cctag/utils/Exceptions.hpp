/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

// This fix is necessary on Apple and on Windows using cygwin to avoid the compilation error
// #error "Boost.Stacktrace requires `_Unwind_Backtrace` function.
// see https://github.com/boostorg/stacktrace/issues/88
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <boost/exception/diagnostic_information.hpp>
#include <boost/exception/errinfo_file_name.hpp>
#include <boost/exception/exception.hpp>
#include <boost/exception/get_error_info.hpp>
#include <boost/exception/info.hpp>
#include <boost/stacktrace.hpp>
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

	error_info() = default;

	error_info( const This& v )
	{
		_value._v << v._value._v.str();
	}

	template<typename V>
	explicit error_info( const V& value )
	{
		_value._v << value;
	}

	~error_info() override = default;

	error_info_base * clone() const override
	{
		return new error_info(*this);
	}

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
	inline std::string name_value_string() const override
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

typedef boost::error_info<struct tag_stacktrace, boost::stacktrace::stacktrace> traced;


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
using user = ::boost::error_info<struct tag_userMessage, ::boost::error_info_sstream>;

/**
 * @brief This is detailed informations for developpers.
 * Not always a real human readable message :)
 * @remark Dev information.
 */
//using dev = ::boost::error_info<struct tag_message,std::string>;
using dev = ::boost::error_info<struct tag_devMessage, ::boost::error_info_sstream>;
//using dev = ::boost::error_info_sstream<struct tag_message>;

/**
 * @brief The algorithm
 * @remark Dev information.
 */
using algorithm = ::boost::error_info<struct tag_algorithm, ::std::string>;
/**
 * @brief Time.
 * @remark Dev or user information.
 */
using time = ::boost::error_info<struct tag_time, float>;
/**
 * @brief Frame number.
 * @remark Dev or user information.
 */
using frame = ::boost::error_info<struct tag_frame, long int>;
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
	File() = default;
	explicit File( const std::string& path )
	{
		*this << filename(path);
	}
};

/**
 * @brief File doesn't exists.
 */
struct FileNotExist : virtual public File
{
	FileNotExist() = default;
	explicit FileNotExist( const std::string& path )
	: File( path )
	{}
};

/**
 * @brief Directory doesn't exists.
 */
struct NoDirectory : virtual public File
{
	NoDirectory() = default;
	explicit NoDirectory( const std::string& path )
	: File( path )
	{}
};

/**
 * @brief Read only file.
 */
struct ReadOnlyFile : virtual public File
{
	ReadOnlyFile() = default;
	explicit ReadOnlyFile( const std::string& path )
	: File( path )
	{}
};
/// @}

}
}
