#ifndef _CCTAG_MARKERS_TYPES_HPP_
#define _CCTAG_MARKERS_TYPES_HPP_

#include "EdgePoint.hpp"

#include <boost/multi_array.hpp>
#include <boost/unordered/unordered_map.hpp>

#include <list>


namespace cctag {
namespace vision {

/**
 * @brief An image (2D array) of pointers to EdgePoints. For each pixel we associate an EdgePoint.
 */
typedef boost::multi_array<EdgePoint*, 2> EdgePointsImage;

typedef boost::unordered_map< EdgePoint*, std::list< EdgePoint* > > WinnerMap;  ///< associate a winner with its voters

}
}

#endif
