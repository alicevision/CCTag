/* 
 * File:   miscellaneous.hpp
 * Author: lcalvet
 *
 * Created on 13 août 2014, 12:27
 */

#ifndef MISCELLANEOUS_HPP
#define	MISCELLANEOUS_HPP

namespace cctag
{


                // todo@Lilian : tout templater

                void coutVPoint(std::list<EdgePoint*> vPoint, int format = 0) {
                    std::cout << "X = [ ";

                    BOOST_FOREACH(const EdgePoint* point, vPoint) {
                        if (format == 0) {
                            std::cout << " [" << point->x() << "," << point->y() << "] ;";
                        } else if (format == 1) {
                            CCTAG_COUT(*point);
                        }
                    }
                    std::cout << "]" << std::endl;
                }

                void coutVPoint(std::vector<EdgePoint*> vPoint, int format = 0) {
                    std::cout << "X = [ ";

                    BOOST_FOREACH(const EdgePoint* point, vPoint) {
                        if (format == 0) {
                            std::cout << " [" << point->x() << "," << point->y() << "] ;";
                        } else if (format == 1) {
                            CCTAG_COUT(*point);
                        }
                    }
                    std::cout << "]" << std::endl;
                }

                void coutVPoint(std::vector<cctag::Point2dN<double> > vPoint) {
#ifdef DEBUG
                    std::cout << "X = [ ";

                    BOOST_FOREACH(const cctag::Point2dN<double> & point, vPoint) {
                        std::cout << " [" << point.x() << "," << point.y() << "] ;";
                    }
                    std::cout << "]" << std::endl;
#endif
                }

} // namespace cctag

#endif	/* MISCELLANEOUS_HPP */

