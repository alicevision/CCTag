#ifndef _CCTAG_CCTAGFLOWCOMPONENT_HPP
#define	_CCTAG_CCTAGFLOWCOMPONENT_HPP

#include "EdgePoint.hpp"

#include "geometry/Ellipse.hpp"

#include <boost/foreach.hpp>
#include <vector>

namespace rom {
    namespace vision {
        namespace marker {

            class CCTagFlowComponent {
            public:
                CCTagFlowComponent();

                CCTagFlowComponent(const std::vector<EdgePoint*> & outerEllipsePoints,
                const std::list<EdgePoint*> & childrens,
                const std::vector<EdgePoint*> & filteredChildrens,
                const rom::numerical::geometry::Ellipse & outerEllipse,
                const std::list<EdgePoint*> & convexEdgeSegment,
                const EdgePoint & seed,
                std::size_t nCircles);
                
                virtual ~CCTagFlowComponent();
                
                void setFieldLines(const std::list<EdgePoint*> & childrens);
                void setFilteredFieldLines(const std::vector<EdgePoint*> & filteredChildrens);

                std::vector<EdgePoint> _outerEllipsePoints;
                rom::numerical::geometry::Ellipse _outerEllipse;
                std::vector<std::vector<EdgePoint> > _fieldLines;
                std::vector<std::vector<EdgePoint> > _filteredFieldLines;
                std::list<EdgePoint> _convexEdgeSegment;
                EdgePoint _seed;
                std::size_t _nCircles;
                
            };
        }
    }
}

#endif	/* CCTAGFLOWCOMPONENT_HPP */
