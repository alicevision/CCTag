#include <math.h>
#include <iostream>

using namespace std;

int main()
{
    int elems = 7;

    cerr << "I have " << elems << " elements in my matrix" << endl;
    int numPairs = elems * ( elems-1 ) / 2;
    cerr << "that makes " << numPairs << " pairs" << endl;

    // 1:0
    // 2:0 2:1
    // 3:0 3:1 3:2
    // 4:0 4:1 4:2 4:3

    cerr << "looking at pairs:" << endl;
    for( int myPair=0; myPair<numPairs; myPair++ ) {

        int j      = (int)( 1+sqrt(1+8*myPair) ) / 2;
        int i      = myPair - j*(j-1)/2;
        cerr << "(" << j << "," << i << ") ";
    }
    cerr << endl;
}

