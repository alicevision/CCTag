#ifndef _TERRY_SAMPLER_HPP_
#define _TERRY_SAMPLER_HPP_

namespace terry {
namespace sampler {

enum EParamFilter
{
	eParamFilterNearest = 0,
	eParamFilterBilinear,
	eParamFilterBC,
	eParamFilterBicubic,
	eParamFilterCatrom,
	eParamFilterKeys,
	eParamFilterSimon,
	eParamFilterRifman,
	eParamFilterMitchell,
	eParamFilterParzen,
	eParamFilterLanczos,
	eParamFilterLanczos3,
	eParamFilterLanczos4,
	eParamFilterLanczos6,
	eParamFilterLanczos12,
	eParamFilterGaussian
};

enum EParamFilterOutOfImage
{
	eParamFilterOutBlack = 0,
	eParamFilterOutTransparency,
	eParamFilterOutCopy,
	eParamFilterOutMirror
};

}
}

#endif

