#ifndef _ROM_CCTAG_HPP_
#define _ROM_CCTAG_HPP_

#include "modeConfig.hpp"

#include "types.hpp"
#include "ellipse.hpp"
#include "Candidate.hpp"
#include "CCTagFlowComponent.hpp"

#include "geometry/point.hpp"
#include "algebra/matrix/Matrix.hpp"
#include "algebra/invert.hpp"
#include "geometry/Ellipse.hpp"
#include "types.hpp"
#include "IOrientedMarker.hpp"

#include "viewGeometry/2DTransform.hpp"
#include "global.hpp"

#include <opencv2/core/types_c.h>

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/array.hpp>
#include <boost/math/constants/constants.hpp>
//#include <boost/numeric/ublas/matrix.hpp>
//#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/fwd.hpp>
#include <boost/ptr_container/ptr_list.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/throw_exception.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

namespace rom {
    namespace vision {
        namespace marker {

            typedef int MarkerID;
            typedef std::vector< std::pair< MarkerID, double > > IdSet;

            namespace ublas = boost::numeric::ublas;
            namespace numerical = rom::numerical;

            class CCTag : public IOrientedMarker {
            public:
                typedef boost::ptr_vector<CCTag> Vector;
                typedef boost::ptr_list<CCTag> List;

                using IMarker::_centerImg;

            public:
                // TODO

                CCTag()
                : _id(0)
                , _quality(0) 
                , _status(0) {
                    setInitRadius();
                }

                CCTag(const MarkerID id,
                        const rom::Point2dN<double> & centerImg,
                        const std::vector< std::vector< Point2dN<double> > > & points,
                        const rom::numerical::geometry::Ellipse & outerEllipse,
                        const rom::numerical::BoundedMatrix3x3d & homography,
                        int pyramidLevel,
                        double scale,
                        const double quality = 1.0)
                : IOrientedMarker(centerImg)
                , _id(id)
                , _outerEllipse(outerEllipse)
                , _points(points)
                , _mHomography(homography)
                , _quality(quality)
                , _pyramidLevel(pyramidLevel)
                , _scale(scale) {
                    setInitRadius();
                    rom::numerical::geometry::scale(_outerEllipse, _rescaledOuterEllipse, scale);
                    _status = 0;
                }

                CCTag(const CCTag & cctag)
                : IOrientedMarker(cctag)
                , _nCircles(cctag._nCircles)
                , _radiusRatios(cctag._radiusRatios)
                , _id(cctag._id)
                , _outerEllipse(cctag._outerEllipse)
                , _ellipses(cctag._ellipses)
                , _points(cctag._points)
                , _mHomography(cctag._mHomography)
                , _quality(cctag._quality)
                , _pyramidLevel(cctag._pyramidLevel)
                , _scale(cctag._scale)
                , _rescaledOuterEllipse(cctag._rescaledOuterEllipse)
                , _status(cctag._status)
#ifdef CCTAG_STAT_DEBUG
                , _flowComponents(cctag._flowComponents)
#endif
                {
                }

                virtual ~CCTag() {
                }

                void scale(const double s);

                std::size_t nCircles() {
                    return _nCircles;
                }

                const rom::numerical::BoundedMatrix3x3d & homography() const {
                    return _mHomography;
                }

                rom::numerical::BoundedMatrix3x3d & homography() {
                    return _mHomography;
                }

                void setHomography(const rom::numerical::BoundedMatrix3x3d & homography) {
                    _mHomography = homography;
                }

                const rom::numerical::geometry::Ellipse & outerEllipse() const {
                    return _outerEllipse;
                }
                
                const std::vector< Point2dN<double> > & rescaledOuterEllipsePoints() const{
                    return _rescaledOuterEllipsePoints;
                }

                const std::vector< std::vector< Point2dN<double> > >& points() const {
                    return _points;
                }

                static const boost::array<double, 5> & radiusRatiosInit() {
                    return _radiusRatiosInit;
                }

                const std::vector<double>& radiusRatios() const {
                    return _radiusRatios;
                }

                std::vector<double> & radiusRatios() {
                    return _radiusRatios;
                }

                void setRadiusRatios(const std::vector<double> radiusRatios) {
                    _radiusRatios = radiusRatios;
                }

                double quality() const {
                    return _quality;
                }

                void setQuality(const double quality) {
                    _quality = quality;
                }

                double scale() const {
                    return _scale;
                }

                void setScale(const double scale) {
                    _scale = scale;
                }

                int pyramidLevel() const {
                    return _pyramidLevel;
                }

                void setPyramidLevel(const int pyramidLevel) {
                    _pyramidLevel = pyramidLevel;
                }

                std::vector<rom::numerical::geometry::Ellipse> & ellipses() {
                    return _ellipses;
                }

                std::vector<rom::numerical::geometry::Ellipse> ellipses() const {
                    return _ellipses;
                }

                void setEllipses(const std::vector<rom::numerical::geometry::Ellipse> ellipses) {
                    _ellipses = ellipses;
                }

                void setRescaledOuterEllipse(const rom::numerical::geometry::Ellipse & rescaledOuterEllipse) {
                    _rescaledOuterEllipse = rescaledOuterEllipse;
                }
                
                void setRescaledOuterEllipsePoints(const std::vector< Point2dN<double> > & outerEllipsePoints){
                    _rescaledOuterEllipsePoints = outerEllipsePoints;
                }

                const rom::numerical::geometry::Ellipse & rescaledOuterEllipse() const {
                    return _rescaledOuterEllipse;
                }

                bool hasId() const {
                    return true;
                }

                MarkerID id() const {
                    return _id;
                }

                void setId(const MarkerID id) {
                    _id = id;
                }

                IdSet idSet() const {
                    return _idSet;
                }

                void setIdSet(const IdSet idSet) {
                    _idSet = idSet;
                }
                
                int getStatus() const{
                    return _status;
                }
                
                
                void setStatus(int status){
                    _status = status;
                }

                bool operator<(const CCTag & tag2) const {
                    return _id < tag2.id();
                }

                //friend std::ostream& operator<<(std::ostream& os, const CCTag& cm);

                inline CCTag* clone() const {
                    return new CCTag(*this);
                }

                void condition(const rom::numerical::BoundedMatrix3x3d & mT, const rom::numerical::BoundedMatrix3x3d & mInvT);

                bool equal(const CCTag& marker) const {
                    return cctag::isSameEllipses(_rescaledOuterEllipse, marker.rescaledOuterEllipse());
                }
                
                

#ifdef CCTAG_STAT_DEBUG

                void addFlowComponent(const cctag::Candidate & candidate) {
                    _flowComponents.push_back(CCTagFlowComponent(candidate._outerEllipsePoints, candidate._childrens, candidate._filteredChildrens, candidate._outerEllipse, candidate._convexEdgeSegment, *(candidate._seed), _nCircles));
                }

                void setFlowComponents(const std::vector<CCTagFlowComponent> & flowComponents) {
                    _flowComponents = flowComponents;
                }

                void setFlowComponents(const std::vector<cctag::Candidate> & candidates) {

                    BOOST_FOREACH(const cctag::Candidate & candidate, candidates) {
                        addFlowComponent(candidate);
                    }
                }

                const std::vector<CCTagFlowComponent> & getFlowComponents() const {
                    return _flowComponents;
                }
#endif

                void serialize(boost::archive::text_oarchive & ar, const unsigned int version);

            protected:

                void setInitRadius() {
                    // todo@Lilian : to be replaced by calling the CCTag bank built from the textfile
                    _radiusRatios.resize(_radiusRatiosInit.size());
                    std::copy(_radiusRatiosInit.begin(), _radiusRatiosInit.end(), _radiusRatios.begin());
                    _nCircles = _radiusRatiosInit.size() + 1;
                }

            protected:
                static const boost::array<double, 5> _radiusRatiosInit;
                std::vector<double> _radiusRatios;

                std::size_t _nCircles;
                MarkerID _id;
                IdSet _idSet;
                rom::numerical::geometry::Ellipse _outerEllipse;
                rom::numerical::geometry::Ellipse _rescaledOuterEllipse;
                std::vector< Point2dN<double> > _rescaledOuterEllipsePoints;
                std::vector<rom::numerical::geometry::Ellipse> _ellipses;
                std::vector< std::vector< Point2dN<double> > > _points;
                rom::numerical::BoundedMatrix3x3d _mHomography;
                double _quality;
                int _pyramidLevel;
                double _scale;
                int _status;

#ifdef CCTAG_STAT_DEBUG
                std::vector<CCTagFlowComponent> _flowComponents;
#endif
            };

            inline CCTag* new_clone(const CCTag& x) {
                return x.clone();
            }
            
            // List of possible status
            static const int no_collected_cuts = -1;
            static const int no_selected_cuts = -2;
            static const int opti_has_diverged = -3;
            static const int id_not_reliable = -4;
            static const int id_reliable = 1;
            
        }
    }
}

#endif
