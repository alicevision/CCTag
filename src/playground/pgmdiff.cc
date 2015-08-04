#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>

using namespace std;

void usage( const char* cmd, string info )
{
    cerr << "Error: " << info << endl
         << endl
         << "Usage: " << cmd << " <outfile> <infile1> <infile2>" << endl
         << "    create a DIFF pgm image outfile from input PGM files infile1 and infile2" << endl
         << endl;
    exit( -1 );
}

int main( int argc, char*argv[] )
{
    if( argc != 4 ) {
        usage( argv[0], "Wrong parameter count" );
    }

    boost::filesystem::path input_file_1( argv[2] );
    boost::filesystem::path input_file_2( argv[3] );

    if( not boost::filesystem::exists( input_file_1 ) )
        usage( argv[0], string("File ") + argv[2] + string( " does not exist" ) );
    if( not boost::filesystem::exists( input_file_2 ) )
        usage( argv[0], string("File ") + argv[3] + string( " does not exist" ) );

    ifstream if1( argv[2] );
    if( not if1.is_open() )
        usage( argv[0], string("File ") + argv[2] + string( " could not be opened for reading" ) );

    ifstream if2( argv[3] );
    if( not if2.is_open() )
        usage( argv[0], string("File ") + argv[3] + string( " could not be opened for reading" ) );

    string pgmtype;
    if1 >> pgmtype;
    if( if1.fail() )
        usage( argv[0], string("File ") + argv[2] + string( " is too short" ) );
    if( pgmtype != "P5" )
        usage( argv[0], string("File ") + argv[2] + string( " does not have type P5" ) );
    if2 >> pgmtype;
    if( if2.fail() )
        usage( argv[0], string("File ") + argv[3] + string( " is too short" ) );
    if( pgmtype != "P5" )
        usage( argv[0], string("File ") + argv[3] + string( " does not have type P5" ) );

    int    w1, h1, w2, h2;
    if1 >> w1 >> h1;
    if( if1.fail() )
        usage( argv[0], string("File ") + argv[2] + string( " is too short" ) );
    if2 >> w2 >> h2;
    if( if2.fail() )
        usage( argv[0], string("File ") + argv[3] + string( " is too short" ) );

    if( w1 <= 0 || h1 <= 0 )
        usage( argv[0], string("File ") + argv[2] + string( " has meaningless image size" ) );
    if( w2 <= 0 || h2 <= 0 )
        usage( argv[0], string("File ") + argv[3] + string( " has meaningless image size" ) );
    if( w1 != w2 || h1 != h2 )
        usage( argv[0], string("Files ") + argv[2] + string(" and " ) + argv[3] +  string( " have different size" ) );

    int    pxsz;
    if1 >> pxsz;
    if( if1.fail() )
        usage( argv[0], string("File ") + argv[2] + string( " is too short" ) );
    if( pxsz != 255 )
        usage( argv[0], string("File ") + argv[2] + string( " is not single-byte encoded" ) );
    if2 >> pxsz;
    if( if2.fail() )
        usage( argv[0], string("File ") + argv[3] + string( " is too short" ) );
    if( pxsz != 255 )
        usage( argv[0], string("File ") + argv[3] + string( " is not single-byte encoded" ) );

    cout << "Image size is " << w1 << "x" << h1 << endl;

    unsigned char* data_in1 = new unsigned char[ w1 * h1 ];
    unsigned char* data_in2 = new unsigned char[ w1 * h1 ];
    if1.read( (char*)data_in1, w1 * h1 );
    if( if1.fail() )
        usage( argv[0], string("File ") + argv[2] + string( " contains too few bytes" ) );
    if2.read( (char*)data_in2, w1 * h1 );
    if( if2.fail() )
        usage( argv[0], string("File ") + argv[3] + string( " contains too few bytes" ) );

    ofstream of(argv[1] );
    if( not of.is_open() )
        usage( argv[0], string("File ") + argv[1] + string( " could not be opened for writing" ) );

    of << "P5" << endl
       << w1 << " " << h1 << endl
       << pxsz << endl;

    unsigned char* data_out = new unsigned char[ w1 * h1 ];
    uint32_t different = 0;

    for( int i=0; i<w1*h1; i++ ) {
        data_out[i] = (unsigned char) abs( (int)data_in1[i] - (int)data_in2[i] );
        if( data_out[i] != 0 ) different += 1;
    }
    of.write( (const char*)data_out, w1*h1 );
    
    if( different == 0 ) {
        cout << "Files " << argv[1] << " and " << argv[2] << " are identical" << endl;
    } else {
        cout << "Files " << argv[1] << " and " << argv[2] << " differ in " << different << " bytes" << endl;
    }
}

