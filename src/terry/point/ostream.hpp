#ifndef _TERRY_POINT_OSTREAM_HPP_
#define _TERRY_POINT_OSTREAM_HPP_


template <typename T>
std::ostream& operator<<( std::ostream& out, const boost::gil::point2<T>& p )
{
	return out << "x:" << p.x << " y:" << p.y;
}


#endif
