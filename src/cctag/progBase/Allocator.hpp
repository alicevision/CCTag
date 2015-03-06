#ifndef _CCTAG_ALLOCATOR_HPP_
#define	_CCTAG_ALLOCATOR_HPP_

#include "exceptions.hpp"

#include <boost/numeric/ublas/vector.hpp>

#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <limits>

namespace cctag {

template<typename T>
class AlignedAllocator
{
public:
	static const std::size_t _alignment = 16;

	// type definitions
	typedef AlignedAllocator<T> This;
	typedef T value_type;
	typedef T * pointer;
	typedef const T * const_pointer;
	typedef T & reference;
	typedef const T & const_reference;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;

	// rebind AlignedAllocator to type U
	template<typename U>
	struct rebind
	{
		typedef AlignedAllocator<U> other;
	};

	// return address of values
	pointer address( reference value ) const
	{
		return &value;
	}

	const_pointer address( const_reference value ) const
	{
		return &value;
	}

	/* constructors and destructor
	 * - nothing to do because the AlignedAllocator has no state
	 */
	AlignedAllocator( ) throw( ) { }

	AlignedAllocator( const AlignedAllocator & ) throw( ) { }

	template<typename U>
	AlignedAllocator( const AlignedAllocator<U> & ) throw( ) { }

	~AlignedAllocator( ) throw( ) { }

	/** return maximum number of elements that can be allocated */
	size_type max_size( ) const throw( )
	{
		return std::numeric_limits<std::size_t>::max( ) / sizeof(T );
	}

	/** allocate but don't initialize num elements of type T */
	pointer allocate( size_type num, const void* hint = 0 )
	{
		void* result = NULL;
		const int ret = posix_memalign( &result, This::_alignment, num*sizeof(T) );
		if( ret != 0 )
		{
#if defined(EINVAL)
			if( ret == EINVAL )
			{
				ROM_THROW( exception::BadAlloc()
					<< exception::dev() + "The alignment argument, value=" /*+ This::_alignment +*/ ", was not a power of two, or was not a multiple of sizeof(void *)." );
			}
#endif
#if defined(ENOMEM)
			if( ret == ENOMEM )
			{
				ROM_THROW( exception::BadAlloc()
					<< exception::dev("Out of memory") );
			}
#endif
			ROM_THROW( exception::BadAlloc()
				<< exception::dev() + "Unrecognized error: " + ret );
		}
		return reinterpret_cast<pointer>( result );
	}

	/** initialize elements of allocated storage p with value value */
	void construct( pointer p, const T &value )
	{
		// initialize memory with placement new
		new( reinterpret_cast < void * > ( p ) ) T( value );
	}

	/** destroy elements of initialized storage p */
	void destroy( pointer p )
	{
		// destroy objects by calling their destructor
		p->~T( );
	}

	/** deallocate storage p of deleted elements */
	void deallocate( pointer p, size_type num )
	{
		free( p );
	}
};

/** return that all specializations of this AlignedAllocator are interchangeable */
template<typename T1, class T2>
bool operator==( const AlignedAllocator<T1> &, const AlignedAllocator<T2>& ) throw( )
{
	return true;
}

template<typename T1, class T2>
bool operator!=( const AlignedAllocator<T1>&, const AlignedAllocator<T2>& ) throw( )
{
	return false;
}

}

#endif

