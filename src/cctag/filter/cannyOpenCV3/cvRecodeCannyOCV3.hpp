/* 
 * File:   cvRecodeCannyOCV3.hpp
 * Author: lilian
 *
 * Created on June 11, 2015, 4:49 PM
 */

#if 0

#ifndef CVRECODECANNYOCV3_HPP
#define	CVRECODECANNYOCV3_HPP

#include "precomp.hpp"

namespace cv
{

void RecodedCanny( InputArray _src, OutputArray _dst,
                double low_thresh, double high_thresh,
                int aperture_size=3, bool L2gradient=false );
}

#endif	/* CVRECODECANNYOCV3_HPP */

#endif

