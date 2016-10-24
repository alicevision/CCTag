#include <iostream>

#include <netpbm/pgm.h>

using namespace std;

int main( )
{
    std::cout << "reading image" << std::endl;

    FILE* f = fopen( "edge_image_1280x720.pgm", "r");

    int cols, rows;
    gray maxval;
    gray** ptr = pgm_readpgm( f, &cols, &rows, &maxval );

    for( int row = 0; row < rows; ++row ) {
        for( int column = 0; column < cols; ++column ) {
            std::cout << ptr[row][column] << " ";
        }
        std::cout << std::endl;
    }

    pgm_freearray( ptr, rows );
    fclose(f);
}

