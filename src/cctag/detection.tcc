#include "CCTagMarkersBank.hpp"

#include "canny.hpp"
#include "vote.hpp"
#include "ellipse.hpp"
#include "identification.hpp"
#include "multiresolution.hpp"
#include "params.hpp"
#include "fileDebug.hpp"

#include "draw.hpp"
#include "cvDraw.hpp"

#include <cctag/toolbox.hpp>

#include <cctag/progBase/MemoryPool.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/image.hpp>
#include <cctag/filter/cvRecode.hpp>
#include <cctag/toolbox/gilTools.hpp>
#include <cctag/algebra/invert.hpp>
#include <cctag/CCTagMarkersBank.hpp>
#include <cctag/statistic/statistic.hpp>
#include <cctag/geometry/distance.hpp>
#include <cctag/geometry/Cercle.hpp>
#include <cctag/boostCv/cvImage.hpp>

//#include <boost/gil/extension/io/jpeg_io.hpp>

#include <boost/gil/image.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/typedefs.hpp>

#include <boost/timer.hpp>

namespace rom {
    namespace vision {
        namespace marker {

            template<class SView>
            void cctagDetection(CCTag::List& markers, const FrameId frame, const SView& srcView, const cctag::Parameters & params, const bool bDisplayEllipses) {
                using namespace cctag;
                using namespace boost::numeric::ublas;
                using namespace boost::gil;
                //	using namespace rom::img;

                std::srand(1);

                static const CCTagMarkersBank bank(params._cctagBankFilename);

                boost::posix_time::ptime tstart(boost::posix_time::microsec_clock::local_time());

                // Grayscale transform
                gray8_image_t grayImg;
                gray8_view_t graySrc = rom::img::toGray(srcView, grayImg);

                // Compute canny
                typedef kth_channel_view_type<0, rgb32f_view_t>::type CannyView;
                typedef kth_channel_view_type<1, rgb32f_view_t>::type GradXView;
                typedef kth_channel_view_type<2, rgb32f_view_t>::type GradYView;

                rgb32f_image_t cannyRGBImg(graySrc.width(), graySrc.height());
                rgb32f_view_t cannyRGB(view(cannyRGBImg));
                CannyView cannyView;
                GradXView cannyGradX;
                GradYView cannyGradY;

                cannyView = kth_channel_view<0>(cannyRGB);
                // x gradient
                cannyGradX = kth_channel_view<1>(cannyRGB);
                // y gradient
                cannyGradY = kth_channel_view<2>(cannyRGB);

                cctag::cannyCv(graySrc, cannyRGB, cannyView, cannyGradX, cannyGradY, params._cannyThrLow, params._cannyThrHigh);

                cctagMultiresDetection(markers, graySrc, cannyRGB, frame, params);

                boost::posix_time::ptime tstop(boost::posix_time::microsec_clock::local_time());

                boost::posix_time::time_duration d = tstop - tstart;
                const double spendTime = d.total_milliseconds();
                ROM_COUT("Etape de detection" << spendTime);

                ////////// Identification pass
                // To decomment -- enable cuts selection, homography computation and identification
                if (1) {
                    ROM_COUT_DEBUG("Before identify!");
                    CCTag::List::iterator it = markers.begin();
                    while (it != markers.end()) {
                        CCTag & cctag = *it;

                        //tstart( boost::posix_time::microsec_clock::local_time() );
                        const int detected = identify(cctag, bank.getMarkers(), graySrc, cannyGradX, cannyGradY, params._numCrowns, params._numCutsInIdentStep, params._numSamplesOuterEdgePointsRefinement, params._cutsSelectionTrials, params._sampleCutLength, params._minIdentProba, params._useLMDif);

                        cctag.setStatus(detected);

                        //if ( detected < 0 )
                        //{
                        //	// Erase marker we were unable to indentify
                        //	it = markers.erase( it );
                        //}
                        //else
                        //{
                        try {
                            std::vector<rom::numerical::geometry::Ellipse> & ellipses = cctag.ellipses();

                            bounded_matrix<double, 3, 3> mInvH;
                            rom::numerical::invert(cctag.homography(), mInvH);

                            BOOST_FOREACH(double radiusRatio, cctag.radiusRatios()) {
                                //ROM_COUT_VAR_DEBUG(radiusRatio);
                                rom::numerical::geometry::Cercle circle(1.0 / radiusRatio);
                                ellipses.push_back(rom::numerical::geometry::Ellipse(prec_prod(trans(mInvH), prec_prod<bounded_matrix<double, 3, 3> >(circle.matrix(), mInvH))));
                            }

                            // Push the outer ellipse
                            ellipses.push_back(cctag.rescaledOuterEllipse());

                            //ROM_COUT_VAR_DEBUG( cctag.ellipses().size() );

                            ROM_COUT_VAR_DEBUG(cctag.id());
                            ++it;
                        } catch (...) {
                            // Impossible to construct Ellipse from computed homographies => conics are not ellipses!
                            //it = markers.erase( it );
                        }
                        //}
                    }
                }

                markers.sort();

                CCTagVisualDebug::instance().writeIdentificationView(markers);

                CCTagFileDebug::instance().newSession("identification.txt");

                BOOST_FOREACH(const CCTag & marker, markers) {
                    CCTagFileDebug::instance().outputMarkerInfos(marker);
                }

                //CCTagVisualDebug::instance().initBackgroundImage(srcView);
                //CCTagVisualDebug::instance().newSession("Identification");
                //
                //BOOST_FOREACH(const CCTag & marker, markers) {
                //    CCTagVisualDebug::instance().drawMarker(marker, false);
                //    CCTagVisualDebug::instance().drawInfos(marker, false);
                //}

                //CCTagVisualDebug::instance().outPutAllSessions();

            }


        }
    }
}

