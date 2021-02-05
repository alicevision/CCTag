#ifndef _CCTAG_ACCUMULATOR
#define _CCTAG_ACCUMULATOR

#include <vector>

namespace cctag
{

class VarianceAccumulator
{
    std::vector<float> acc;

public:
    VarianceAccumulator( ) { }
    void insert( float f );
    float result( ) const;
};

class MeanAccumulator
{
    std::vector<float> acc;

public:
    MeanAccumulator( ) { }
    void insert( float f );
    float result( ) const;
};

class LMeanAccumulator
{
    std::vector<long> acc;

public:
    LMeanAccumulator( ) { }
    void insert( long f );
    long result( ) const;
};

}; // namespace cctag

#endif

