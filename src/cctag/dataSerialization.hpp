#ifndef _ROM_DATASERIALIZATION_HPP
#define	_ROM_DATASERIALIZATION_HPP

#include "CCTagFlowComponent.hpp"
#include <cctag/CCTag.hpp>

//#include <rom/vision/feature/marker/IMarker.hpp>
#include "geometry/Ellipse.hpp"
#include "algebra/matrix/Matrix.hpp"

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/foreach.hpp>

namespace rom {
    namespace vision {
        namespace marker {

            void serializeRadiusRatios(boost::archive::text_oarchive & ar, const std::vector<double> & radiusRatios);

            void serializeIdSet(boost::archive::text_oarchive & ar, const IdSet & idSet);

            void serializePoint(boost::archive::text_oarchive & ar, const Point2dN<double> & point);
            
            void serializeEdgePoint(boost::archive::text_oarchive & ar, const EdgePoint & e);

            void serializeVecPoint(boost::archive::text_oarchive & ar, const std::vector< Point2dN<double> > & points);

            void serializePoints(boost::archive::text_oarchive & ar, const std::vector< std::vector< Point2dN<double> > > & points);

            void serializeEllipse(boost::archive::text_oarchive & ar, const rom::numerical::geometry::Ellipse & ellipse);

            void serializeEllipses(boost::archive::text_oarchive & ar, const std::vector<rom::numerical::geometry::Ellipse> & ellipses);

            void serializeBoundedMatrix3x3d(boost::archive::text_oarchive & ar, const rom::numerical::BoundedMatrix3x3d & matrix);

            void serializeFlowComponent(boost::archive::text_oarchive & ar, const CCTagFlowComponent & flowComponent);

            void serializeFlowComponents(boost::archive::text_oarchive & ar, const std::vector<CCTagFlowComponent> & flowComponents);
            
        }
    }
}

#endif	/* SERIALIZATION_HPP */

