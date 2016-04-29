#ifndef VISION_MARKER_CCTAG_MARKERS_BANK_HPP
#define	VISION_MARKER_CCTAG_MARKERS_BANK_HPP

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

namespace cctag
{

class CCTagMarkersBank
{
public:
  CCTagMarkersBank( const std::size_t nCrowns );
  CCTagMarkersBank( const std::string & file );
  
  virtual ~CCTagMarkersBank();

  void read( const std::string & file );
  std::size_t identify( const std::vector<float> & marker ) const;
  inline const std::vector< std::vector<float> > & getMarkers() const
  {
    return _markers;
  }

private:
  template <typename Iterator>
  bool cctagLineParse( Iterator first, Iterator last, std::vector<float>& rr )
  {
    float n;
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
  
  static const float idThreeCrowns[32][5];
  static const float idFourCrowns[128][7];
  
  std::vector< std::vector<float> > _markers;

};

} // namespace cctag

#endif
