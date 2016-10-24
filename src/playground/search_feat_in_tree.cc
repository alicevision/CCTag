#include <iostream>

struct Feature
{
    float dim[128];

    Feature( bool randomly )
    {
        if( randomly )
            for( int i=0; i<128; i++ ) {
                dim[i] = drand48();
            }
        else
            for( int i=0; i<128; i++ ) {
                dim[i] = 0;
            }
    }
};

struct cmp_by_dim
{
    int _index;
    cmp_by_dim( int i ) : _index(i) { }

    bool operator<()( float l, float r ) {
        return l.dim[i] < r.dim[i];
    }
};

struct cmp_to_cut
{
    int index;
    float value;
    cmp_to_cut( int i, float limit ) : index(i), value(limit) { }
    bool operator()( float v ) {
        return v >= value;
    }
};

int main( )
{
    const int howmany = 1000000;
    const int cmpwith = 500;

    vector<float> cuts( 128 );
    for( int i=0; i<128; i++ ) {
        cuts[i] = drand48();
    }

    vector<Feature> f;
    f.reserve( howmany );
    for( int i=0; i<howmany; i++ ) f.push_back( Feature( true ) );

    vector<Feature> c;
    c.reserve( cmpwith );
    for( int i=0; i<cmpwith; i++ ) c.push_back( Feature( true ) );

    vector<Feature>::const_iterator cut_iterators[128][128][2];
    vector<Feature>::const_iterator it;

    cut_iterators[0][0][0] = f.begin();
    cut_iterators[0][0][1] = f.end();

    for( int line=0; line<128; line++ ) {
        for( int j=0; j<line; j++ ) {
            cmp_by_dim comparator(line);
            std::sort( cut_iterators[line][j][0], cut_iterators[line][j][1], comparator );
        }

        for( int j=0; j<line; j++ ) {
            cmp_to_cut comparator( line, cuts[line] );
            it = f.find_first_of( comparator );
            if( line < 127 ) {
                cut_iterators[line+1][j*2  ][0] = cut_iterators[line][j][0];
                cut_iterators[line+1][j*2  ][1] = it;
                cut_iterators[line+1][j*2+1][0] = it;
                cut_iterators[line+1][j*2+1][1] = cut_iterators[line][j][1];
            }
        }
    }
}

