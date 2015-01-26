#include "CCTagMarkersBank.hpp"
#include <cctag/progBase/exceptions.hpp>
#include <cctag/global.hpp>

#include <boost/numeric/conversion/bounds.hpp>
#include <boost/throw_exception.hpp>

#include <cmath>
#include <fstream>
#include <vector>

namespace rom {
namespace vision {
namespace marker {

CCTagMarkersBank::CCTagMarkersBank( const std::string & file )
{
	read( file );
}

CCTagMarkersBank::~CCTagMarkersBank()
{
}

void CCTagMarkersBank::read( const std::string & file )
{
	std::ifstream input( file.c_str() );
	if ( !input.good() )
	{
		BOOST_THROW_EXCEPTION( exception::Value()
			<< exception::dev() + "Unable to open bank file: " + file );
	}
	std::string str;
	while ( std::getline( input, str ) )
	{
		std::vector<double> rr;
		cctagLineParse( str.begin(), str.end(), rr);
		if ( rr.size() )
		{
			_markers.push_back( rr );
		}
	}
}

std::size_t CCTagMarkersBank::identify( const std::vector<double> & marker ) const
{
	std::vector< std::vector<double> >::const_iterator itm = _markers.begin();

	std::size_t imin = 0;
	double normMin = boost::numeric::bounds<double>::highest();
	std::size_t i = 0;

	while( itm != _markers.end() )
	{
		std::vector<double>::const_iterator itr1 = marker.begin();
		std::vector<double>::const_iterator itr2 = itm->begin();
		double norm = 0;
		while( itr1 != marker.end() && itr2 != itm->end() )
		{
			norm += ( *itr1 - *itr2 ) * ( *itr1 - *itr2 );
			++itr1;
			++itr2;
		}
		++itm;
		norm = std::sqrt( norm );
		ROM_COUT_LILIAN( "Res : " << norm );
		if ( norm < normMin )
		{
			normMin = norm;
			imin = i;
		}
		++i;
	}

	if ( normMin > 0.6 )
	{
		BOOST_THROW_EXCEPTION( rom::exception::Bug() << rom::exception::dev() + "Unable to identify marker" );
	}
	else
	{
		return imin + 1;
	}
}



}
}
}
