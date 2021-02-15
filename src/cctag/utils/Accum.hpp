#pragma once

#include <vector>

namespace cctag
{

class VarianceAccumulator
{
    std::vector<float> acc;

public:
    VarianceAccumulator() = default;
    void insert( float f );
    float result( ) const;
};

class MeanAccumulator
{
    std::vector<float> acc;

public:
    MeanAccumulator() = default;
    void insert( float f );
    float result( ) const;
};

class LMeanAccumulator
{
    std::vector<long> acc;

public:
    LMeanAccumulator() = default;
    void insert( long f );
    long result( ) const;
};

}; // namespace cctag

