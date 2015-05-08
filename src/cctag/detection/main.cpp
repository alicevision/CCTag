#define png_infopp_NULL (png_infopp)NULL
#define int_p_NULL (int*)NULL
#include <boost/gil/extension/io/png_io.hpp>

#include <cctag/fileDebug.hpp>
#include <cctag/visualDebug.hpp>
#include <cctag/progBase/exceptions.hpp>
#include <cctag/progBase/MemoryPool.hpp>
#include <cctag/detection.hpp>
#include <cctag/view.hpp>
#include <cctag/image.hpp>

#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
#include <boost/progress.hpp>
#include <boost/gil/gil_all.hpp>
#include <boost/gil/image.hpp>
#include <boost/gil/extension/io/jpeg_io.hpp>
#include <boost/exception/all.hpp>
#include <boost/ptr_container/ptr_list.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>

#include <terry/sampler/all.hpp>
#include <terry/sampler/resample_subimage.hpp>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>

#include <sstream>
#include <iostream>
#include <string>
#include <fstream>
#include <exception>

using namespace cctag;
using boost::timer;

using namespace boost::gil;
namespace bfs = boost::filesystem;

static const std::string kUsageString = "Usage: detection image_file.png\n";

void detection(std::size_t frame, cctag::View& view, const cctag::Parameters & params, std::string outputFileName = "")
{
    POP_ENTER;
    
    if (outputFileName == "") {
      outputFileName = "00000";
    }
    
    // Process markers detection
    boost::timer t;
    boost::ptr_list<CCTag> markers;

    view.setNumLayers( params._numberOfMultiresLayers );
    
    CCTagVisualDebug::instance().initBackgroundImage(view._view);
    CCTagVisualDebug::instance().setImageFileName(outputFileName);
    CCTagFileDebug::instance().setPath(CCTagVisualDebug::instance().getPath());
    
    cctagDetection(markers, frame, view._grayView, params, true);
    
    CCTagFileDebug::instance().outPutAllSessions();
    CCTagFileDebug::instance().clearSessions();
    CCTagVisualDebug::instance().outPutAllSessions();
    CCTagVisualDebug::instance().clearSessions();

    CCTAG_COUT( markers.size() << " markers.");
    CCTAG_COUT("Total time: " << t.elapsed());
    CCTAG_COUT_NOENDL("Id : ");

    int i = 0;
    BOOST_FOREACH(const cctag::CCTag & marker, markers) {
        cctag::drawMarkerOnGilImage(view._view, marker, false);
        cctag::drawMarkerInfos(view._view, marker, false);

        if (i == 0) {
            CCTAG_COUT_NOENDL(marker.id() + 1);
        } else {
            CCTAG_COUT_NOENDL(", " << marker.id() + 1);
        }
        ++i;
    }
    CCTAG_COUT("");
    POP_LEAVE;
}

/*************************************************************/
/*                    Main entry                             */
/*************************************************************/
int main(int argc, char** argv)
{
    try {
        if (argc <= 1) {
            BOOST_THROW_EXCEPTION(cctag::exception::Bug() << cctag::exception::user() + kUsageString);
        }
        cctag::MemoryPool::instance().updateMemoryAuthorizedWithRAM();
        const std::string filename(argv[1]);
        
        // Set all the parameters
        std::string paramsFilename;
        if (argc >= 3) {
            paramsFilename = argv[2];
        }
        cctag::Parameters params;
        
        if (paramsFilename != "") {
          // Read the parameter file provided by the user
          std::ifstream ifs(paramsFilename.c_str());
          boost::archive::xml_iarchive ia(ifs);
          ia >> boost::serialization::make_nvp("CCTagsParams", params);
          CCTAG_COUT("Parameters contained in " << paramsFilename << " are used.");
        } else {
          // Use the default parameters and save them in defaultParameters.xml
          paramsFilename = "defaultParameters.xml";
          std::ofstream ofs(paramsFilename.c_str());
          boost::archive::xml_oarchive oa(ofs);
          oa << boost::serialization::make_nvp("CCTagsParams", params);
          CCTAG_COUT("Parameter file not provided. Default parameters are used.");
        }

        CCTagVisualDebug::instance().initializeFolders(filename, params._numCrowns);
        bfs::path myPath(filename);
        std::string ext(myPath.extension().string());
        
        if ( (ext == ".png") || (ext == ".jpg") ) {

          POP_INFO << "looking at image " << myPath.string() << std::endl;
          
          cctag::View my_view( filename );

          rgb8_image_t& image = my_view._image;
          rgb8_view_t&  svw   = my_view._view;

          // Increase image size.
          /*{
                  rgb8_image_t simage;
                  simage.recreate( 2 * image.width(), 2 * image.height() );
                  rgb8_view_t osvw( view( simage ) );
                  terry::resize_view( svw, osvw, terry::sampler::bicubic_sampler() );
          }*/

          // Call the CCTag detection
          detection(0, my_view, params, myPath.stem().string());
          
        } else if (ext == ".avi" )
        {
          CCTAG_COUT("*** Video mode ***");
          POP_INFO << "looking at video " << myPath.string() << std::endl;
          
          // open video and check
          cv::VideoCapture video(filename.c_str());
          if(!video.isOpened())
          {
            CCTAG_COUT("Unable to open the video : " << filename); return -1;
          }

          // play loop
          int lastFrame = video.get(CV_CAP_PROP_FRAME_COUNT);
          int frameId = 0;
          while( video.get(CV_CAP_PROP_POS_FRAMES) < lastFrame )
          {
            cv::Mat frame;
            video >> frame;
            cv::Mat imgGray;
            cv::cvtColor( frame, imgGray, CV_BGR2GRAY );
            cctag::View cctagView((const unsigned char *) imgGray.data, imgGray.cols, imgGray.rows , imgGray.step );
            cctagView._view = boost::gil::interleaved_view(imgGray.cols, imgGray.rows, (boost::gil::rgb8_pixel_t*) frame.data, frame.step );

            // Set the output folder
            std::stringstream outFileName;
            outFileName << std::setfill('0') << std::setw(5) << frameId;
            
            // Call the CCTag detection
            detection(frameId, cctagView, params, outFileName.str());
            
            ++frameId; 
          }
        } else {
            throw std::logic_error("Unrecognized input.");
        }
    } catch (...) {
        std::cerr << boost::current_exception_diagnostic_information() << std::endl;
        return 1;
    }
    return 0;
}

