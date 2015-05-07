#ifndef MISCELLANEOUS_HPP
#define	MISCELLANEOUS_HPP

namespace cctag
{
// todo@Lilian : tout templater

void coutVPoint(std::list<EdgePoint*> vPoint, int format = 0) {
    CCTAG_COUT_NOENDL("X = [ ");

    BOOST_FOREACH(const EdgePoint* point, vPoint) {
        if (format == 0) {
            CCTAG_COUT_NOENDL(" [" << point->x() << "," << point->y() << "] ;");
        } else if (format == 1) {
            CCTAG_COUT(*point);
        }
    }
    CCTAG_COUT("]");
}

void coutVPoint(std::vector<EdgePoint*> vPoint, int format = 0) {
    CCTAG_COUT_NOENDL("X = [ ");

    BOOST_FOREACH(const EdgePoint* point, vPoint) {
        if (format == 0) {
            CCTAG_COUT_NOENDL(" [" << point->x() << "," << point->y() << "] ;");
        } else if (format == 1) {
            CCTAG_COUT(*point);
        }
    }
    CCTAG_COUT("]");
}

void coutVPoint(std::vector<cctag::Point2dN<double> > vPoint) {
#ifdef DEBUG
    CCTAG_COUT_NOENDL("X = [ ");

    BOOST_FOREACH(const cctag::Point2dN<double> & point, vPoint) {
        CCTAG_COUT_NOENDL(" [" << point.x() << "," << point.y() << "] ;");
    }
    CCTAG_COUT("]");
#endif
}

} // namespace cctag

#endif	/* MISCELLANEOUS_HPP */

