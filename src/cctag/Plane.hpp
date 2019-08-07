/*
 * Copyright 2019, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_PLANE_HPP_
#define _CCTAG_PLANE_HPP_

#include <opencv2/opencv.hpp>

namespace cctag {

/*************************************************************
 * PlaneType
 *************************************************************/

template<typename SubType> struct PlaneType
{
    int cvType() { return CV_8UC1; }
};

template<> struct PlaneType<uint8_t>
{
    int cvType() { return CV_8UC1; }
};

template<> struct PlaneType<int16_t>
{
    int cvType() { return CV_16SC1; }
};

template<> struct PlaneType<int32_t>
{
    int cvType() { return CV_32SC1; }
};

/*************************************************************
 * Plane
 *************************************************************/

template<typename Type> class Plane
{
    typedef PlaneType<Type> T;

    Type*  _buffer;
    size_t _height;
    size_t _width;
    int*   _refCount;

public:
    Plane( );
    Plane( Plane& plane );
    Plane( size_t h, size_t w );
    Plane( Type* buffer, size_t h, size_t w );
    ~Plane( );

    Plane& operator=( Plane& plane );

    cv::Mat     getMat( );
    cv::Mat     getMat( )  const;
    size_t      getCols( ) const;
    size_t      getRows( ) const;
    Type*       getBuffer( );
    const Type* getBuffer( ) const;

    Type&       at( int x, int y );
    const Type& at( int x, int y ) const;

    void release();

private:
    void unref();
};

template<typename Type>
Plane<Type>::Plane( )
    : _buffer( 0 )
    , _height( 0 )
    , _width( 0 )
    , _refCount( 0 )
{ }

template<typename Type>
Plane<Type>::Plane( Plane& plane )
    : _buffer( plane._buffer )
    , _height( plane._height )
    , _width(  plane._width  )
    , _refCount( plane._refCount )
{
    if( _refCount ) *_refCount += 1;
}

template<typename Type>
Plane<Type>::Plane( Type* buffer, size_t h, size_t w )
    : _buffer( buffer )
    , _height( h )
    , _width( w )
    , _refCount( 0 )
{ }

template<typename Type>
Plane<Type>::Plane( size_t h, size_t w )
    : _buffer( new Type[h*w] )
    , _height( h )
    , _width( w )
    , _refCount( new int )
{
    *_refCount = 1;
}

template<typename Type>
void Plane<Type>::release( )
{
    unref();
}

template<typename Type>
void Plane<Type>::unref( )
{
    if( _refCount && ( *_refCount > 0 ) )
    {
        *_refCount -= 1;
        if( *_refCount == 0 )
        {
            delete _refCount;
            delete [] _buffer;
        }
    }
    _buffer   = 0;
    _height   = 0;
    _width    = 0;
    _refCount = 0;
}

template<typename Type>
Plane<Type>::~Plane( )
{
    unref();
}

template<typename Type>
Plane<Type>& Plane<Type>::operator=( Plane<Type>& plane )
{
    unref();

    _buffer   = plane._buffer;
    _height   = plane._height;
    _width    = plane._width;
    _refCount = plane._refCount;
    if( _refCount )
    {
        *_refCount += 1;
    }
    return *this;
}

template<typename Type>
Type& Plane<Type>::at( int x, int y )
{
    return _buffer[ y * _width + x ];
}

template<typename Type>
const Type& Plane<Type>::at( int x, int y ) const
{
    return _buffer[ y * _width + x ];
}

template<typename Type>
cv::Mat Plane<Type>::getMat( )
{
    T t;
    cv::Mat mat( _height, _width, t.cvType(), (void*)_buffer );
    return mat;
}

template<typename Type>
cv::Mat Plane<Type>::getMat( ) const
{
    T t;
    cv::Mat mat( _height, _width, t.cvType(), (void*)_buffer );
    return mat;
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
    return _buffer;
}

template<typename Type>
const Type* Plane<Type>::getBuffer( ) const
{
    return _buffer;
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

} // namespace cctag

#endif

