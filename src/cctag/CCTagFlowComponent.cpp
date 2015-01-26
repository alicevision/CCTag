#include "CCTagFlowComponent.hpp"
#include "global.hpp"

namespace rom {
    namespace vision {
        namespace marker {

            CCTagFlowComponent::CCTagFlowComponent() {
            }

            CCTagFlowComponent::CCTagFlowComponent(
                    const std::vector<EdgePoint*> & outerEllipsePoints,
                    const std::list<EdgePoint*> & childrens,
                    const std::vector<EdgePoint*> & filteredChildrens,
                    const rom::numerical::geometry::Ellipse & outerEllipse,
                    const std::list<EdgePoint*> & convexEdgeSegment,
                    const EdgePoint & seed,
                    std::size_t nCircles)
            : _outerEllipse(outerEllipse)
            , _seed(seed) 
            , _nCircles(nCircles)
            {

                _outerEllipsePoints.reserve(outerEllipsePoints.size());

                BOOST_FOREACH(const EdgePoint * e, outerEllipsePoints) {
                    _outerEllipsePoints.push_back(EdgePoint(*e));
                }

                BOOST_FOREACH(const EdgePoint * e, convexEdgeSegment) {
                    _convexEdgeSegment.push_back(EdgePoint(*e));
                }


                setFieldLines(childrens);
                setFilteredFieldLines(filteredChildrens);

            }

            CCTagFlowComponent::~CCTagFlowComponent() {
            }

            // todo@Lilian : templater les 2 methodes suivantes sur le container

            void CCTagFlowComponent::setFilteredFieldLines(const std::vector<EdgePoint*> & filteredChildrens) {

                _filteredFieldLines.resize(filteredChildrens.size());

                std::size_t i = 0;

                for (std::vector<EdgePoint*>::const_iterator it = filteredChildrens.begin(); it != filteredChildrens.end(); ++it) {
                    int dir = -1;
                    EdgePoint* p = *it;

                    std::vector<EdgePoint> & vE = _filteredFieldLines[i];

                    vE.reserve(_nCircles);
                    vE.push_back(EdgePoint(*p));

                    for (std::size_t j = 1; j < _nCircles; ++j) {
                        if (dir == -1) {
                            p = p->_before;
                        } else {
                            p = p->_after;
                        }

                        vE.push_back(EdgePoint(*p));

                        dir = -dir;
                    }
                    ++i;
                }
            }

            void CCTagFlowComponent::setFieldLines(const std::list<EdgePoint*> & childrens) {

                _fieldLines.resize(childrens.size());

                std::size_t i = 0;

                for (std::list<EdgePoint*>::const_iterator it = childrens.begin(); it != childrens.end(); ++it) {
                    int dir = -1;
                    EdgePoint* p = *it;

                    std::vector<EdgePoint> & vE = _fieldLines[i];

                    vE.reserve(_nCircles);
                    vE.push_back(EdgePoint(*p));

                    for (std::size_t j = 1; j < _nCircles; ++j) {
                        if (dir == -1) {
                            p = p->_before;
                        } else {
                            p = p->_after;
                        }

                        vE.push_back(EdgePoint(*p));

                        dir = -dir;
                    }
                    ++i;
                }
            }
        }
    }
}
