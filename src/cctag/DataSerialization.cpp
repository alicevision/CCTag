#include <cctag/DataSerialization.hpp>

namespace cctag {

void serializeRadiusRatios(boost::archive::text_oarchive & ar, const std::vector<double> & radiusRatios) {
    const int sizeRadiusRatios = radiusRatios.size();
    ar & BOOST_SERIALIZATION_NVP(sizeRadiusRatios);

    BOOST_FOREACH(const double & ratio, radiusRatios) {
        ar & BOOST_SERIALIZATION_NVP(ratio);
    }
}

void serializeIdSet(boost::archive::text_oarchive & ar, const IdSet & idSet) {
    const int sizeIdSet = idSet.size();
    ar & BOOST_SERIALIZATION_NVP(sizeIdSet);

    typedef std::pair< MarkerID, double > IdPair;

    BOOST_FOREACH(const IdPair & idPair, idSet) {
        ar & BOOST_SERIALIZATION_NVP(idPair.first);
        ar & BOOST_SERIALIZATION_NVP(idPair.second);
    }
}

void serializePoint(boost::archive::text_oarchive & ar, const Point2d<Eigen::Vector3f> & point) {
    const double x = point.x();
    const double y = point.y();

    ar & BOOST_SERIALIZATION_NVP(x);
    ar & BOOST_SERIALIZATION_NVP(y);
}

// todo templater function above and bellow.
void serializePoint(boost::archive::text_oarchive & ar, const DirectedPoint2d<Eigen::Vector3f> & point) {
    const double x = point.x();
    const double y = point.y();

    ar & BOOST_SERIALIZATION_NVP(x);
    ar & BOOST_SERIALIZATION_NVP(y);
}

void serializeEdgePoint(boost::archive::text_oarchive & ar, const EdgePoint & e) {
    const int x = e.x();
    const int y = e.y();
    const double gx = e.gradient()(0);
    const double gy = e.gradient()(1);

    ar & BOOST_SERIALIZATION_NVP(x);
    ar & BOOST_SERIALIZATION_NVP(y);
    ar & BOOST_SERIALIZATION_NVP(gx);
    ar & BOOST_SERIALIZATION_NVP(gy);
}

void serializeVecPoint(boost::archive::text_oarchive & ar, const std::vector< DirectedPoint2d<Eigen::Vector3f> > & points) {
    const int sizePoints = points.size();
    ar & BOOST_SERIALIZATION_NVP(sizePoints);

    BOOST_FOREACH(const DirectedPoint2d<Eigen::Vector3f> & point, points) {
        serializePoint(ar, point);
    }
}

void serializePoints(boost::archive::text_oarchive & ar, const std::vector< std::vector< DirectedPoint2d<Eigen::Vector3f> > > & points) {
    const int sizePoints = points.size();
    ar & BOOST_SERIALIZATION_NVP(sizePoints);

    BOOST_FOREACH(const std::vector< DirectedPoint2d<Eigen::Vector3f> > & subPoints, points) {
        serializeVecPoint(ar, subPoints);
    }
}

void serializeEllipse(boost::archive::text_oarchive & ar, const cctag::numerical::geometry::Ellipse & ellipse) {
    serializeBoundedMatrix3x3d(ar, ellipse.matrix());
}

void serializeEllipses(boost::archive::text_oarchive & ar, const std::vector<cctag::numerical::geometry::Ellipse> & ellipses) {
    const int sizeEllipses = ellipses.size();
    ar & BOOST_SERIALIZATION_NVP(sizeEllipses);

    BOOST_FOREACH(const cctag::numerical::geometry::Ellipse & ellipse, ellipses) {
        serializeEllipse(ar, ellipse);
    }
}

void serializeBoundedMatrix3x3d(boost::archive::text_oarchive & ar, const Eigen::Matrix3f & matrix) {
    ar & BOOST_SERIALIZATION_NVP(matrix(0, 0));
    ar & BOOST_SERIALIZATION_NVP(matrix(1, 0));
    ar & BOOST_SERIALIZATION_NVP(matrix(2, 0));
    ar & BOOST_SERIALIZATION_NVP(matrix(0, 1));
    ar & BOOST_SERIALIZATION_NVP(matrix(1, 1));
    ar & BOOST_SERIALIZATION_NVP(matrix(2, 1));
    ar & BOOST_SERIALIZATION_NVP(matrix(0, 2));
    ar & BOOST_SERIALIZATION_NVP(matrix(1, 2));
    ar & BOOST_SERIALIZATION_NVP(matrix(2, 2));
}

void serializeFlowComponent(boost::archive::text_oarchive & ar, const CCTagFlowComponent & flowComponent) {
    const std::vector<EdgePoint> & outerEllipsePoints = flowComponent._outerEllipsePoints;

    const std::size_t sizeOuterEllipsePoints = outerEllipsePoints.size();
    ar & BOOST_SERIALIZATION_NVP(sizeOuterEllipsePoints);

    BOOST_FOREACH(const EdgePoint & e, outerEllipsePoints) {
        serializeEdgePoint(ar, e);
    }

    serializeEllipse(ar, flowComponent._outerEllipse);

    const std::size_t sizeFilteredChildrens = flowComponent._filteredFieldLines.size();
    ar & BOOST_SERIALIZATION_NVP(sizeFilteredChildrens);

    BOOST_FOREACH(const std::vector<EdgePoint> & fL, flowComponent._filteredFieldLines) {

        BOOST_FOREACH(const EdgePoint & e, fL) {
            serializeEdgePoint(ar, e);
        }
    }

    const std::size_t sizeChildrens = flowComponent._fieldLines.size();
    ar & BOOST_SERIALIZATION_NVP(sizeChildrens);

    BOOST_FOREACH(const std::vector<EdgePoint> & fL, flowComponent._fieldLines) {

        BOOST_FOREACH(const EdgePoint & e, fL) {
            serializeEdgePoint(ar, e);
        }
    }

    const std::list<EdgePoint> & convexEdgeSegment = flowComponent._convexEdgeSegment;
    const std::size_t sizeConvexEdgeSegment = convexEdgeSegment.size();
    ar & BOOST_SERIALIZATION_NVP(sizeConvexEdgeSegment);

    BOOST_FOREACH(const EdgePoint & e, convexEdgeSegment) {
        serializeEdgePoint(ar, e);
    }

    serializeEdgePoint(ar, flowComponent._seed);
}

void serializeFlowComponents(boost::archive::text_oarchive & ar, const std::vector<CCTagFlowComponent> & flowComponents) {

    const std::size_t sizeFlowComponents = flowComponents.size();
    ar & BOOST_SERIALIZATION_NVP(sizeFlowComponents);

    BOOST_FOREACH(const CCTagFlowComponent & flowComponent, flowComponents) {
        serializeFlowComponent(ar, flowComponent);
    }
}

} // namespace cctag
