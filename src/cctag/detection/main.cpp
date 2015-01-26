
#define png_infopp_NULL (png_infopp)NULL
#define int_p_NULL (int*)NULL
#include <boost/gil/extension/io/png_io.hpp>


#include <cctag/fileDebug.hpp>
#include <cctag/visualDebug.hpp>

//#include <rom/engine/RomConfig.hpp>
//#include <cctag/graphics/image.hpp>
//#include <rom/engine/io/ffmpeg/VideoFFmpegReader.hpp>
//#include <rom/engine/io/ffmpeg/VideoFFmpegWriter.hpp>

//#include <rom/engine/processing/OMMarkerDetection.hpp>

#include <cctag/progBase/exceptions.hpp>
#include <cctag/detection.hpp>

#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
#include <boost/progress.hpp>
#include <boost/gil/gil_all.hpp>
#include <boost/gil/image.hpp>
#include <boost/exception/all.hpp>
#include <boost/ptr_container/ptr_list.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>

#include <terry/sampler/all.hpp>
#include <terry/sampler/resample_subimage.hpp>

#include <sstream>
#include <iostream>
#include <string>
#include <fstream>
#include <exception>

//#include <boost/gil/extension/io/jpeg_io.hpp>

//using namespace rom::graphics;
using namespace rom::vision;
using boost::timer;

using namespace boost::gil;
namespace bfs = boost::filesystem;

static const std::string kUsageString = "Usage: detection image_file.png\n";

void detection(rom::FrameId frame, rgb8_view_t& view, const std::string & paramsFilename = "") {
    // Process markers detection
    boost::timer t;
    boost::ptr_list<marker::CCTag> markers;
    rom::vision::marker::cctag::Parameters params;
    if (paramsFilename != "") {
        std::ifstream ifs(paramsFilename.c_str());
        boost::archive::xml_iarchive ia(ifs);
        // write class instance to archive
        ia >> boost::serialization::make_nvp("CCTagsParams", params);
    } else {
        std::ofstream ofs("detectionCCTagParams.xml");
        boost::archive::xml_oarchive oa(ofs);
        // write class instance to archive
        oa << boost::serialization::make_nvp("CCTagsParams", params);
    }
    
    cctagDetection( markers, frame, view, params, true );
    
    //rom::processing::markerDetection(frame, view, markers, &params, true);

    std::cout << "Id : ";

    int i = 0;

    //CCTagVisualDebug::instance().newSession( "Identification" );

    BOOST_FOREACH(const rom::vision::marker::CCTag & marker, markers) {
        rom::vision::marker::drawMarkerOnGilImage(view, marker, false);
        rom::vision::marker::drawMarkerInfos(view, marker, false);

        //CCTagVisualDebug::instance().drawMarker( marker, false);
        //CCTagVisualDebug::instance().drawInfos( marker, false);

        if (i == 0) {
            std::cout << marker.id() + 1;
        } else {
            std::cout << ", " << marker.id() + 1;
        }
        ++i;
    }
    std::cout << std::endl;

    ROM_TCOUT("Detected: " << markers.size() << " markers.");
    ROM_TCOUT("CCDetection global time: " << t.elapsed());
}

/*************************************************************/
/*                    Main entry                             */

/*************************************************************/
int main(int argc, char** argv) {
    try {
        if (argc <= 1) {
            BOOST_THROW_EXCEPTION(rom::exception::Bug() << rom::exception::user() + kUsageString);
        }
        rom::MemoryPool::instance().updateMemoryAuthorizedWithRAM();
        const std::string filename(argv[1]);
        std::string paramsFilename;
        if (argc >= 3) {
            paramsFilename = argv[2];
        }

        bfs::path myPath(filename);

        const bfs::path extPath(myPath.extension()); // ( bfs::extension( filename ) );

        const bfs::path subFilenamePath(myPath.filename());

        const bfs::path parentPath(myPath.parent_path());


        std::string ext(extPath.string());

        std::ofstream stream;
        stream.open("dataHomographies.txt");
        stream.close();

        /*
                        std::ofstream oms;
                        stream.open("points.txt");
                        stream.close();
         */

        bfs::path folder(argv[1]);

        if ((ext == ".png") || (ext == ".jpg")) {
            rgb8_image_t image;
            rgb8_view_t svw;
            if (ext == ".png") {
                image = rgb8_image_t(png_read_dimensions(filename.c_str()));
                svw = rgb8_view_t(view(image));
                png_read_and_convert_view(filename.c_str(), svw);
            } else {
                image = rgb8_image_t(jpeg_read_dimensions(filename.c_str()));
                svw = rgb8_view_t(view(image));
                jpeg_read_and_convert_view(filename.c_str(), svw);
            }

            // Increase image size.
            /*{
                    rgb8_image_t simage;
                    simage.recreate( 2 * image.width(), 2 * image.height() );
                    rgb8_view_t osvw( view( simage ) );
                    terry::resize_view( svw, osvw, terry::sampler::bicubic_sampler() );
            }*/

            // Create inputImagePath/result if it does not exist
            std::stringstream resultFolderName, localizationFolderName, identificationFolderName;
            resultFolderName << parentPath.string() << "/result";
            localizationFolderName << resultFolderName.str() << "/localization";
            identificationFolderName << resultFolderName.str() << "/identification";

            if (!bfs::exists(resultFolderName.str())) {
                bfs::create_directory(resultFolderName.str());
            }

            if (!bfs::exists(localizationFolderName.str())) {
                bfs::create_directory(localizationFolderName.str());
            }

            if (!bfs::exists(identificationFolderName.str())) {
                bfs::create_directory(identificationFolderName.str());
            }

            // Set the output image name, holding the same name as the original image
            // so that the result will be placed in the inputImagePath/result/ folder
            std::stringstream strstreamResult;
            strstreamResult << resultFolderName.str() << "/" << myPath.stem().string() << ".png";
            ROM_TCOUT(strstreamResult.str());

            resultFolderName << "/" << myPath.stem().string();

            if (!bfs::exists(resultFolderName.str())) {
                bfs::create_directory(resultFolderName.str());
            }

            CCTagVisualDebug::instance().initBackgroundImage(svw);
            CCTagVisualDebug::instance().initPath(resultFolderName.str());
            CCTagVisualDebug::instance().setImageFileName(myPath.stem().string());

            CCTagFileDebug::instance().setPath(resultFolderName.str());

            detection(0, svw, paramsFilename);

            CCTagFileDebug::instance().outPutAllSessions();
            CCTagFileDebug::instance().clearSessions();

            // write visual file output
            CCTagVisualDebug::instance().outPutAllSessions();
        } else if (bfs::is_directory(folder)) {
            std::cout << folder << " is a directory containing:\n";

            std::vector<bfs::path> filesInFolder;

            copy(bfs::directory_iterator(folder), bfs::directory_iterator(), back_inserter(filesInFolder)); // is directory_entry, which is
            // converted to a path by the
            // path stream inserter

            sort(filesInFolder.begin(), filesInFolder.end());

            std::size_t frame = 0;

            BOOST_FOREACH(const bfs::path & fileInFolder, filesInFolder) {
                ROM_TCOUT(fileInFolder);

                const std::string ext(bfs::extension(fileInFolder));

                std::stringstream outFilename, inFilename;

                if (ext == ".png") {
                    inFilename << argv[3] << "/orig/" << std::setfill('0') << std::setw(5) << frame << ".png";
                    outFilename << argv[3] << "/" << std::setfill('0') << std::setw(5) << frame << ".png";

                    rgb8_image_t image(png_read_dimensions(fileInFolder.c_str()));
                    rgb8_view_t sourceView(view(image));
                    png_read_and_convert_view(fileInFolder.c_str(), sourceView);

                    png_write_view(inFilename.str(), sourceView);

                    CCTagVisualDebug::instance().initBackgroundImage(sourceView);

                    detection(frame, sourceView, paramsFilename);

                    png_write_view(outFilename.str(), sourceView);

                    ++frame;
                }
            }

            //for (std::vector<bfs::path>::const_iterator it (filesInFolder.begin()); it != filesInFolder.end(); ++it)
            //{
            //  std::cout << "   " << *it << '\n';
            //}

        } else {
            throw std::logic_error("Unrecognized input.");
        }
    } catch (...) {
        std::cerr << boost::current_exception_diagnostic_information() << std::endl;
        return 1;
    }
    return 0;
}

