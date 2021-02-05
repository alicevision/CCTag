#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include <cctag/utils/Accum.hpp>

namespace bacc  = boost::accumulators;

namespace cctag
{

void VarianceAccumulator::insert( float f )
{
    acc.push_back( f );
}

float VarianceAccumulator::result( ) const
{
    bacc::accumulator_set< float, bacc::features< bacc::tag::variance > > v;
    for( auto f : acc ) v.operator()(f);
    return bacc::variance( v );
}

void MeanAccumulator::insert( float f )
{
    acc.push_back( f );
}

float MeanAccumulator::result( ) const
{
    bacc::accumulator_set< float, bacc::features< bacc::tag::mean > > v;
    for( auto f : acc ) v.operator()(f);
    return bacc::mean( v );
}

void LMeanAccumulator::insert( long f )
{
    acc.push_back( f );
}

long LMeanAccumulator::result( ) const
{
    bacc::accumulator_set< long,  bacc::features< bacc::tag::mean > > v;
    for( auto f : acc ) v.operator()(f);
    return bacc::mean( v );
}

}; // namespace cctag

