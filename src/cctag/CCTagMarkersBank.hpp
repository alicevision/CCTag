#ifndef _CCTAG_VISION_CCTAGMARKERSIO_HPP_
#define	_CCTAG_VISION_CCTAGMARKERSIO_HPP_

#include <cctag/global.hpp>

#include <boost/function.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/spirit/include/qi.hpp>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace rom {
namespace vision {
namespace marker {

class CCTagMarkersBank
{
public:
	CCTagMarkersBank( const std::string & file );
	virtual ~CCTagMarkersBank();

	void read( const std::string & file );
	std::size_t identify( const std::vector<double> & marker ) const;
	inline const std::vector< std::vector<double> > & getMarkers() const
	{
		return _markers;
	}

private:
    template <typename Iterator>
    bool cctagLineParse( Iterator first, Iterator last, std::vector<double>& rr )
    {
		double n;
		using boost::phoenix::ref;
		using boost::phoenix::push_back;
		using namespace boost::spirit::qi;
		using boost::spirit::qi::_1;
        bool r = phrase_parse( first, last,
            //  Begin grammar
            (
                *( ( (uint_[ boost::phoenix::ref( n ) = _1 ] >> '/' >> uint_[ boost::phoenix::ref( n ) = boost::phoenix::ref( n ) / _1 ]) | double_[ boost::phoenix::ref( n ) = _1 ] )[ push_back( boost::phoenix::ref(rr), boost::phoenix::ref( n ) ) ] )
            )
            ,
            //  End grammar
            space );

        if ( first != last )
		{
			// fail if we did not get a full match
            return false;
		}
        return r;
    }

private:
	std::vector< std::vector<double> > _markers;

};

}
}
}

#endif
