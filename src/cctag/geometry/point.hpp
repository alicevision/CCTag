#ifndef _CCTAG_POINT_HPP_
#define	_CCTAG_POINT_HPP_

#include <cctag/progBase/exceptions.hpp>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/functional/hash.hpp>

namespace cctag {


/**
 * @brief A Homogeneous 2D point [x, y, w].
 */
template<class T>
class Point2dH : public boost::numeric::ublas::bounded_vector<T, 3>
{
public:
	typedef boost::numeric::ublas::bounded_vector<T, 3> Container;
	typedef Container Parent;
	typedef Point2dH<T> This;

	template<class V>
	Point2dH( const Point2dH<V>& p )
		: Parent( 3 )
	{
		using boost::numeric_cast;
		this->setX( numeric_cast<T>( p.x() ) );
		this->setY( numeric_cast<T>( p.y() ) );
		this->setW( numeric_cast<T>( p.w() ) );
	}

	Point2dH()
		: Parent( 3 )
	{
		this->setX( 0 );
		this->setY( 0 );
		this->setW( 1 );
	}

	Point2dH( const This& p )
		: Parent( 3 )
	{
		this->setX( p.x() );
		this->setY( p.y() );
		this->setW( p.w() );
	}

	Point2dH( const T px, const T py )
		: Parent( 3 )
	{
		this->setX( px );
		this->setY( py );
		this->setW( 1 );
	}

	Point2dH( const Container& p )
		: Parent( 3 )
	{
		this->setX( p(0) );
		this->setY( p(1) );
		this->setW( p(2) );
	}

	virtual ~Point2dH() {}

//	virtual cv::Point_<T> cvPoint() const
//	{
//		cv::Point_<T> p;
//		p.x = x() / w();
//		p.y = y() / w();
//		return p;
//	}

	inline This& toNonHomogen()
	{
		if( w() == 0 )
		{
			BOOST_THROW_EXCEPTION( exception::Bug()
				<< exception::dev() + "Normalization of an infinite point !" );
		}
		x() /= w();
		y() /= w();
		this->setW( 1.0 );
		return *this;
	}

	template<class P>
	This& operator=( const P& p )
	{
		using boost::numeric_cast;
		this->setX( numeric_cast<T>( p.x() ) );
		this->setY( numeric_cast<T>( p.y() ) );
		this->setW( numeric_cast<T>( p.w() ) );
		return *this;
	}

	bool operator==( const This& p ) const
	{
		return
			x() == p.x() &&
			y() == p.y() &&
			w() == p.w();
	}

	virtual boost::numeric::ublas::bounded_vector<T, 2> getNormalizedBoundedVec2d() const
	{
		boost::numeric::ublas::bounded_vector<T, 2> v( 2 );
		if( w() == 0 )
		{
			v( 0 ) = x();
			v( 1 ) = y();
			return v;
		}
		v( 0 ) = x() / w();
		v( 1 ) = y() / w();
		return v;
	}

	inline T  x() const { return (*this)( 0 ); }
	inline T& x()       { return (*this)( 0 ); }

	inline T  y() const { return (*this)( 1 ); }
	inline T& y()       { return (*this)( 1 ); }

	inline T  w() const { return (*this)( 2 ); }
	inline T& w()       { return (*this)( 2 ); }

	inline T    getX() const { return x(); }
	inline T    getY() const { return y(); }
	inline T    getW() const { return w(); }
	inline void setX( const T px ) { x() = px; }
	inline void setY( const T py ) { y() = py; }
	inline void setW( const T pw ) { w() = pw; }

	inline const Container& getContainer() const { return *this; }
	inline Container& getContainer() { return *this; }
};

/**
 * @brief A 2D normalized point [x, y, 1].
 */
template<class T>
class Point2dN : public Point2dH<T>
{
	typedef Point2dH<T> Parent;
	typedef typename Parent::Container Container;
	typedef Point2dN<T> This;

public:
	Point2dN()
		: Parent()
	{}

	Point2dN( const Container& p )
		: Parent(p)
	{
		Parent::toNonHomogen();
	}

	Point2dN( const This& p )
		: Parent( p.x(), p.y() )
	{}

	Point2dN( const T px, const T py )
		: Parent( px, py )
	{}

	virtual ~Point2dN()
	{}

//	cv::Point_<T> cvPoint() const
//	{
//		cv::Point_<T> p;
//		p.x = Parent::x();
//		p.y = Parent::y();
//		return p;
//	}

	template<typename TT>
	This& operator=( const Point2dN<TT>& p )
	{
		using boost::numeric_cast;
		this->setX( numeric_cast<T>( p.x() ) );
		this->setY( numeric_cast<T>( p.y() ) );
		this->setW( (T)1 );
		return *this;
	}
	This& operator=( const Point2dH<T>& p )
	{
		if( p.w() == 0 )
		{
			this->setX( p.x() );
			this->setY( p.y() );
			this->setW( (T)1 );
			return *this;
		}
		this->setX( p.x() / p.w() );
		this->setY( p.y() / p.w() );
		this->setW( (T)1 );
		return *this;
	}
};

/**
 * @brief A 2D normalized point [x, y, 1] + its gradient bounded_vector<T,2>
 */
template<class T>
class DirectedPoint2d : public Point2dN<T>
{
	typedef Point2dN<T> Parent;
	typedef DirectedPoint2d<T> This;

public:
	DirectedPoint2d()
		: Parent()
	{
          _grad.clear();   
        }
                
        DirectedPoint2d( const This& p )
		: Parent(p.x(), p.y())
	{
          _grad(0) = p.dX();
          _grad(1) = p.dY();
        }

	DirectedPoint2d( const Parent& p, const T dX, const T dY)
		: Parent( p.x(), p.y() )
	{
          _grad(0) = dX;
          _grad(1) = dY;
        }

	DirectedPoint2d( const T px, const T py, const T dX, const T dY)
		: Parent( px, py )
	{
          _grad(0) = dX;
          _grad(1) = dY;
        }
        
        inline T  dX() const { return _grad(0); }
        inline T  dY() const {return  _grad(1); }
        
        inline void  setDX(T dX ) { _grad(0) = dX; }
        inline void  setDY(T dY ) { _grad(1) = dY; }
        
        const boost::numeric::ublas::bounded_vector<T,2> & gradient() const 
        {
          return _grad;
        }

	virtual ~DirectedPoint2d()
	{}
        
private:
        boost::numeric::ublas::bounded_vector<T,2> _grad;
};

struct UnHomogenizer
{
	template<class T>
	void operator()( T& p ) { p.toNonHomogen(); }
};

template<class T>
struct Point3d : public boost::numeric::ublas::bounded_vector<T, 3>
{
	Point3d( const boost::numeric::ublas::bounded_vector<T, 3> & p )
		: boost::numeric::ublas::bounded_vector<T, 3>( p )
	{
	}

	Point3d()
		: boost::numeric::ublas::bounded_vector<T, 3>( 3 )
	{
		this->setX( 0 );
		this->setY( 0 );
		this->setZ( 0 );
	}

	Point3d( const T px, const T py, const T pz )
		: boost::numeric::ublas::bounded_vector<T, 3>( 3 )
	{
		this->setX( px );
		this->setY( py );
		this->setZ( pz );
	}

	inline T  x() const { return (*this)( 0 ); }
	inline T& x()       { return (*this)( 0 ); }

	inline T  y() const { return (*this)( 1 ); }
	inline T& y()       { return (*this)( 1 ); }

	inline T  z() const { return (*this)( 2 ); }
	inline T& z()       { return (*this)( 2 ); }

	inline void setX( const T px ) { x() = px; }
	inline void setY( const T py ) { y() = py; }
	inline void setZ( const T pz ) { z() = pz; }

	inline void setNormalized( const T x, const T y, const T z, const T w )
	{
		if( w == 0 )
		{
			this->setX( 0 );
			this->setY( 0 );
			this->setZ( 0 );
			return;
		}
		this->setX( x / w );
		this->setY( y / w );
		this->setZ( z / w );
	}

	inline void setNormalized( const boost::numeric::ublas::bounded_vector<T, 4>& p )
	{
		if( p[3] == 0 )
		{
			this->setX( 0 );
			this->setY( 0 );
			this->setZ( 0 );
			return;
		}
		this->setX( p[0]/p[3] );
		this->setY( p[1]/p[3] );
		this->setZ( p[2]/p[3] );
	}
};

template<class T>
inline size_t hash_value( const Point2dH<T> & v )
{
	size_t h = 0xdeadbeef;
	boost::hash_combine( h, v.x() );
	boost::hash_combine( h, v.y() );
	boost::hash_combine( h, v.w() );
	return h;
}

}

#endif

