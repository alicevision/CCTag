/*
 * Copyright 2019, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_PLANE_HPP_
#define _CCTAG_PLANE_HPP_

#include <cstdint>
#include <string>
#include <memory>
#include <vector>

#include "cctag/Colors.hpp"

namespace cctag {

/*************************************************************
 * A set of PlaneDesctructor types
 *************************************************************/

template<typename Type>
struct HstDestructorNotMine
{
    void operator()( Type* ) { }
};

template<typename Type>
struct HstDestructorMine
{
    void operator()( Type* ptr) { delete [] ptr; }
};

/*************************************************************
 * Plane
 *************************************************************/

template<typename Type> class Plane
{
    std::shared_ptr<Type> _buffer;
    size_t                _height;
    size_t                _width;

public:
    Plane( );
    Plane( const Plane& plane );
    Plane( size_t h, size_t w );
    Plane( Type* buffer, size_t h, size_t w );
    ~Plane( );

    Plane& operator=( const Plane& plane );

    Plane clone( ) const;

    void reset( )
    {
        _buffer.reset();
        _height = 0;
        _width  = 0;
    }

    size_t      getCols( ) const;
    size_t      getRows( ) const;
    Type*       getBuffer( );
    const Type* getBuffer( ) const;

    Type&       at( int x, int y );
    const Type& at( int x, int y ) const;
};

template<typename Type>
Plane<Type>::Plane( )
    : _buffer( 0, HstDestructorNotMine<Type>() )
    , _height( 0 )
    , _width( 0 )
{ }

template<typename Type>
Plane<Type>::Plane( const Plane& plane )
    : _buffer( plane._buffer )
    , _height( plane._height )
    , _width(  plane._width  )
{ }

template<typename Type>
Plane<Type>::Plane( Type* buffer, size_t h, size_t w )
    : _buffer( buffer, HstDestructorNotMine<Type>() )
    , _height( h )
    , _width( w )
{ }

template<typename Type>
Plane<Type>::Plane( size_t h, size_t w )
    : _buffer( new Type[h*w], HstDestructorMine<Type>() )
    , _height( h )
    , _width( w )
{
}

template<typename Type>
Plane<Type>::~Plane( )
{ }

template<typename Type>
Plane<Type>& Plane<Type>::operator=( const Plane<Type>& plane )
{
    _buffer   = plane._buffer;
    _height   = plane._height;
    _width    = plane._width;
    return *this;
}

template<typename Type>
Plane<Type> Plane<Type>::clone( ) const
{
    int h = this->getRows();
    int w = this->getCols();
    Plane<Type> dest( h, w );
    memcpy( dest.getBuffer(), this->getBuffer(), h*w*sizeof(Type) );
    return dest;
}

template<typename Type>
Type& Plane<Type>::at( int x, int y )
{
    Type* p = _buffer.get();
    return p[ y * _width + x ];
}

template<typename Type>
const Type& Plane<Type>::at( int x, int y ) const
{
    const Type* p = _buffer.get();
    return p[ y * _width + x ];
}

template<typename Type>
size_t Plane<Type>::getCols( ) const
{
    return _width;
}

template<typename Type>
size_t Plane<Type>::getRows( ) const
{
    return _height;
}

template<typename Type>
Type*   Plane<Type>::getBuffer( )
{
    return _buffer.get();
}

template<typename Type>
const Type* Plane<Type>::getBuffer( ) const
{
    return _buffer.get();
}

/*************************************************************
 * writePlane
 *************************************************************/

#define SCALED_WRITING   true
#define UNSCALED_WRITING false

/* create a file with the given name and print a PGM image of
 * type P5 (user-readable).
 * If scaled is true, normalize the values in the plane between
 * 0 and 255. If scaled is false, cut values to [0..255]
 */
void writePlanePGM( const std::string&     filename,
                    const Plane<int8_t>&   plane,
                    bool                   scaled );
void writePlanePGM( const std::string&     filename,
                    const Plane<uint8_t>&  plane,
                    bool                   scaled );
void writePlanePGM( const std::string&     filename,
                    const Plane<int16_t>&  plane,
                    bool                   scaled );
void writePlanePGM( const std::string&     filename,
                    const Plane<uint16_t>& plane,
                    bool                   scaled );
void writePlanePPM( const std::string&     filename,
                    const Plane<Color>&    plane,
                    bool                   scaled );

} // namespace cctag

#endif

