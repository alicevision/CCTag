#ifndef _TERRY_FREETYPE_UTIL_STL_HPP_
#define _TERRY_FREETYPE_UTIL_STL_HPP_

// (C) Copyright Tom Brinkman 2007.
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt.)

namespace std
{

template<typename _InIt, typename _InIt2, typename Pred>
inline
_InIt find_if( _InIt _First, _InIt _Last, _InIt2 _First2, Pred pred )
{
	for(; _First != _Last; ++_First, ++_First2 )
		if( pred( *_First, *_First2 ) )
			break;
	return ( _First );
}

template<typename _InIt, typename _InIt2, typename _Fn1>
inline
_Fn1 for_each( _InIt _First, _InIt _Last, _InIt2 _First2, _Fn1 _Func )
{
	for(; _First != _Last; ++_First, ++_First2 )
		_Func( *_First, *_First2 );
	return ( _Func );
}

template<typename _InIt, typename _InIt2, typename _InIt3, typename _Fn1>
inline
_Fn1 for_each( _InIt _First, _InIt _Last, _InIt2 _First2, _InIt3 _First3, _Fn1 _Func )
{
	for(; _First != _Last; ++_First, ++_First2, ++_First3 )
		_Func( *_First, *_First2, *_First3 );
	return ( _Func );
}

template<typename _InIt, typename _InIt2, typename _InIt3, typename _InIt4, typename _Fn1>
inline
_Fn1 for_each( _InIt _First, _InIt _Last, _InIt2 _First2, _InIt3 _First3, _InIt4 _First4, _Fn1 _Func )
{
	for(; _First != _Last; ++_First, ++_First2, ++_First3, ++_First4 )
		_Func( *_First, *_First2, *_First3, *_First4 );
	return ( _Func );
}

}

#endif

