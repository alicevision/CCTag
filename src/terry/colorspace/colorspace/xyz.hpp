#ifndef _TERRY_COLOR_COLORSPACE_XYZ_HPP_
#define	_TERRY_COLOR_COLORSPACE_XYZ_HPP_

#include "base.hpp"

namespace terry {
namespace color {


/// \addtogroup ColorNameModel
/// \{
namespace xyz_colorspace
{
/// \brief 
struct X_t {};    
/// \brief 
struct Y_t {};
/// \brief 
struct Z_t {}; 
}
/// \}

/// \ingroup ColorSpaceModel
typedef ::boost::mpl::vector3<
		xyz_colorspace::X_t,
		xyz_colorspace::Y_t,
		xyz_colorspace::Z_t
	> xyz_colorspace_t;

/// \ingroup LayoutModel
typedef layout<xyz_colorspace_t> xyz_layout_t;

struct XYZParams : public IColorParams
{
	int _xyzValue;
	XYZParams()
	: _xyzValue(567)
	{}
	typedef XYZParams This;
	virtual bool operator==( const IColorParams& otherBase ) const
	{
		const IColorParams* otherBasePtr = &otherBase;
		const This* otherPtr = dynamic_cast<const This*>( otherBasePtr );
		return otherPtr != NULL;
	}
};
/**
 * @brief XYZ colorspace description
 * @todo
 */
struct XYZ
{
	typedef IsRootReference reference;
	typedef XYZParams params;
	
	typedef xyz_colorspace_t colorspace;
	typedef xyz_layout_t layout;
};



}
TERRY_DEFINE_GIL_INTERNALS_3(xyz)
TERRY_DEFINE_COLORSPACE_STANDARD_TYPEDEFS(xyz)
}

#endif

