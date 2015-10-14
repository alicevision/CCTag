/* /////////////////////////////////////////////////////////////////////////////
//
//                  INTEL CORPORATION PROPRIETARY INFORMATION
//     This software is supplied under the terms of a license agreement or
//     nondisclosure agreement with Intel Corporation and may not be copied
//     or disclosed except in accordance with the terms of that agreement.
//          Copyright(c) 2014 Intel Corporation. All Rights Reserved.
//
*/

#if !defined( __IPPICV_TYPES_H__ )
#define __IPPICV_TYPES_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DFTSpec_C_32fc       IppsDFTSpec_C_32fc;
typedef struct DFTSpec_C_64fc       IppsDFTSpec_C_64fc;
typedef struct DFTSpec_R_32f        IppsDFTSpec_R_32f;
typedef struct DFTSpec_R_64f        IppsDFTSpec_R_64f;

struct DCT2DFwdSpec_32f;
typedef struct DCT2DFwdSpec_32f IppiDCTFwdSpec_32f;

struct DCT2DInvSpec_32f;
typedef struct DCT2DInvSpec_32f IppiDCTInvSpec_32f;

struct DFT2DSpec_C_32fc;
typedef struct DFT2DSpec_C_32fc IppiDFTSpec_C_32fc;

struct DFT2DSpec_R_32f;
typedef struct DFT2DSpec_R_32f IppiDFTSpec_R_32f;

typedef enum {
    ippiFilterBilateralGauss = 100,
    ippiFilterBilateralGaussFast = 101
} IppiFilterBilateralType;

struct FilterBilateralSpec;
typedef struct FilterBilateralSpec IppiFilterBilateralSpec;

typedef enum {
    ippDistNormL1   =   0x00000002
} IppiDistanceMethodType;

typedef enum {
    ippNearest = IPPI_INTER_NN,
    ippLinear = IPPI_INTER_LINEAR,
    ippCubic = IPPI_INTER_CUBIC2P_CATMULLROM,
    ippLanczos = IPPI_INTER_LANCZOS,
    ippHahn = 0,
    ippSuper = IPPI_INTER_SUPER
} IppiInterpolationType;

typedef struct ResizeSpec_32f   IppiResizeSpec_32f;
typedef struct ResizeSpec_64f   IppiResizeSpec_64f;

typedef struct {
    Ipp32u borderLeft;
    Ipp32u borderTop;
    Ipp32u borderRight;
    Ipp32u borderBottom;
} IppiBorderSize;

struct ippcvFilterGaussianSpec;
typedef struct ippcvFilterGaussianSpec IppFilterGaussianSpec;

struct ipcvHaarClassifier_32f;
typedef struct ipcvHaarClassifier_32f IppiHaarClassifier_32f;

struct ipcvMorphState;
typedef struct ipcvMorphState IppiMorphState;

struct ipcvMorphAdvState;
typedef struct ipcvMorphAdvState IppiMorphAdvState;

struct ipcvMorphGrayState_8u;
typedef struct ipcvMorphGrayState_8u IppiMorphGrayState_8u;

struct ipcvMorphGrayState_32f;
typedef struct ipcvMorphGrayState_32f IppiMorphGrayState_32f;

typedef struct MomentState64f IppiMomentState_64f;

struct PyramidState;
typedef struct PyramidState IppiPyramidState;

typedef IppiPyramidState IppiPyramidDownState_8u_C1R;
typedef IppiPyramidState IppiPyramidDownState_16u_C1R;
typedef IppiPyramidState IppiPyramidDownState_32f_C1R;
typedef IppiPyramidState IppiPyramidDownState_8u_C3R;
typedef IppiPyramidState IppiPyramidDownState_16u_C3R;
typedef IppiPyramidState IppiPyramidDownState_32f_C3R;
typedef IppiPyramidState IppiPyramidUpState_8u_C1R;
typedef IppiPyramidState IppiPyramidUpState_16u_C1R;
typedef IppiPyramidState IppiPyramidUpState_32f_C1R;
typedef IppiPyramidState IppiPyramidUpState_8u_C3R;
typedef IppiPyramidState IppiPyramidUpState_16u_C3R;
typedef IppiPyramidState IppiPyramidUpState_32f_C3R;


typedef struct _IppiPyramid {
    Ipp8u         **pImage;
    IppiSize      *pRoi;
    Ipp64f        *pRate;
    int           *pStep;
    Ipp8u         *pState;
    int            level;
} IppiPyramid;


typedef enum _IppiKernelType {
    ippKernelSobel     =  0,
    ippKernelScharr    =  1
} IppiKernelType;


typedef enum _IppiNorm {
    ippiNormInf = 0,
    ippiNormL1 = 1,
    ippiNormL2 = 2,
    ippiNormFM = 3
} IppiNorm;


#ifdef __cplusplus
}
#endif

#endif /* __IPPICV_TYPES_H__ */
