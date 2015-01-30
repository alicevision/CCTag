#ifndef _TERRY_COLOR_COLORSPACE_HPP_
#define	_TERRY_COLOR_COLORSPACE_HPP_

#include "colorspace/xyz.hpp"

//#include "layout.hpp"
//#include "gradation.hpp"
//#include "primaries.hpp"
//#include "temperature.hpp"

#include <tuttle/common/utils/global.hpp>

#include <terry/basic_colors.hpp>
#include <terry/numeric/init.hpp>

#include <terry/numeric/operations.hpp>

#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/push_front.hpp>
#include <boost/mpl/pop_back.hpp>
#include <boost/mpl/pop_front.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/transform_view.hpp>
#include <boost/mpl/zip_view.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/print.hpp>
#include <boost/type_traits/is_same.hpp>

#include <boost/ptr_container/ptr_vector.hpp>

#include <boost/numeric/ublas/matrix.hpp>

#include <boost/fusion/mpl.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/adapted/mpl.hpp>
#include <boost/fusion/sequence/intrinsic.hpp>
#include <boost/fusion/algorithm/iteration/for_each.hpp>

#include <boost/assert.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_base_of.hpp>

#include <vector>
#include <typeinfo>
#include <iostream>


namespace terry {

using namespace ::boost::gil;

/*
- static / dynamic => from/to XYZ
- chaque colorspace se défini relativement à un autre
- chaque parametre a sa jeu de paramètre => structure associée
- FullColorParam
*/
namespace color {


namespace {
struct print_type
{
	template<class T>
	void operator( )( T t )
	{
		std::cout << typeid(T).name( ) << '\n';
	}
};
}

/**
ll lib	 * @brief Allows to retrieve an mpl::vector of all color types hierachy.
 * @example for HSL: from_root=[XYZ,RGB,HSL], to_root=[HSL,RGB,XYZ]
 */
template<class color>
struct color_dependencies
{
	/// list of color types dependencies from root (eg. HSL: [XYZ,RGB,HSL])
	typedef typename ::boost::mpl::push_back<typename color_dependencies<typename color::reference>::from_root, color>::type from_root;
	/// list of color types dependencies to root (eg. HSL: [HSL,RGB,XYZ])
	typedef typename ::boost::mpl::push_front<typename color_dependencies<typename color::reference>::to_root, color>::type to_root;
	///< number of color types dependencies (eg. HSL: 3)
	typedef typename ::boost::mpl::size<from_root>::type size;

	struct details
	{
		typedef typename ::boost::mpl::pop_back<from_root>::type from_root_A;
		typedef typename ::boost::mpl::pop_front<from_root>::type from_root_B;
		typedef typename ::boost::mpl::pop_back<to_root>::type to_root_A;
		typedef typename ::boost::mpl::pop_front<to_root>::type to_root_B;
	};
	/// a vector of colors pairs (eg. HSL: [(XYZ,RGB),(RGB,HSL)])
	typedef typename ::boost::mpl::zip_view< ::boost::mpl::vector<typename details::from_root_A, typename details::from_root_B> >::type color_steps_from_root;
	/// a vector of colors pairs (eg. HSL: [(HSL,RGB),(RGB,XYZ)])
	typedef typename ::boost::mpl::zip_view< ::boost::mpl::vector<typename details::to_root_A, typename details::to_root_B> >::type color_steps_to_root;
};
template<>
struct color_dependencies<IsRootReference>
{
	typedef ::boost::mpl::vector<> from_root;
	typedef ::boost::mpl::vector<> to_root;
};


///**
// * @brief Base class of all color basic operations.
// */
//struct IColorOperation
//{
//	typedef IColorOperation This;
//	virtual ~IColorOperation() = 0;
//	
//	virtual bool mergeable() const = 0;
//	virtual void merge( const IColorOperation& op ) const {}
//	
//	void operator()( Pixel& pix ) const = 0;
//	virtual This* clone() const = 0;
//};
//inline IColorOperation* new_clone( const IColorOperation& a )
//{
//	return a.clone();
//}
//
//using boost::numeric::ublas::matrix;
//
//struct MatrixOperation
//{
//	typedef MatrixOperation This;
//	MatrixOperation();
//	
//	bool mergeable() const { return true; }
//	void merge( const IColorOperation& op ) const
//	{
//		using namespace boost::numeric::ublas;
//		const MatrixOperation& matOp = dynamic_cast<const MatrixOperation&>( op );
//		this->_matrix = prod( this->_matrix, matOp._matrix );
//	}
//	template<class Pixel>
//	void operator()( Pixel& pix ) const
//	{
//	}
//	
//	virtual This* clone() const
//	{
//		return new This( *this );
//	}
//private:
//	matrix<double> _matrix;
//};

/**
 * @brief Base class of all color transformation functor.
 * @todo how to template pixels?
 */
struct IColorTransformation
{
	virtual ~IColorTransformation() = 0;
	
	virtual bool mergeable() const = 0;
	virtual void operator()() const = 0;
	virtual IColorTransformation* clone() const = 0;
};
inline IColorTransformation* new_clone( const IColorTransformation& a )
{
	return a.clone();
}

template<class ColorSrc, class ColorDst, bool direction_is_from_root>
struct ColorTransformation : public IColorTransformation
{
	typedef ColorTransformation<ColorSrc, ColorDst, direction_is_from_root> This;

	typedef typename ColorSrc::Params SrcParams;

	typedef boost::mpl::if_c< direction_is_from_root,
		typename ColorDst::Params,
		typename ColorSrc::Params
		> Params;
	
	Params& _params;
	
	ColorTransformation( const Params& params )
	: _params( params )
	{}
	
	template<class PixelSrc, class PixelDst>
	void operator()( const PixelSrc& src, PixelDst& dst ) const
	{
		color_transform( _params, src, dst );
	}
	
	virtual IColorTransformation* clone() const
	{
		return new This( *this );
	}
};

/**
 * @brief Base class for a color hierarchy of parameters.
 */
struct IFullColorParams
{
	virtual std::vector<IColorParams*> getColorspaceVectorView() = 0;
	virtual const std::vector<IColorParams*> getColorspaceVectorView() const = 0;
	
	/**
	 * @brief Color hierarchy size.
	 */
	virtual std::size_t getNbReferences() const = 0;
};


/**
 * @brief Mpl functor to create a dynamic std::vector view on static ::boost::mpl::vector values.
 * The dynamic vector is a list of pointers to the static size ::boost::mpl::vector.
 * 
 * The goal is to work dynamically on a static type. This allows to dynamically
 * compare 2 types.
 */
template<class AbstractClass>
struct append_abstractptr_from_fusionvector_to_stdvector
{
	std::vector<AbstractClass*>& _vec;
	/**
     * @param vec vector to fill.
     */
	append_abstractptr_from_fusionvector_to_stdvector( std::vector<AbstractClass*>& vec )
	: _vec( vec )
	{}
	template<typename T>
	void operator()( T& t ) const
	{
//		BOOST_STATIC_ASSERT(( typename ::boost::is_base_of< AbstractClass, T >::type ));
//		std::cout << typeid(T).name( ) << '\n';
//		std::cout << "&t: " << &t << '\n';
//		std::cout << "t: " << t << '\n';
		_vec.push_back( &t );
	}
};

/**
 * @brief Mpl functor to convert a list of color types into a list of color params types
 * @example HSL>RGB>XYZ to HSLParams>RGBParams>XYZParams
 */
template< typename Color >
struct color_to_colorparams_mplfunc
{
	typedef typename Color::params type;
};

/**
 * @brief All needed parameters to fully defined a specific colorspace.
 * 
 * To fully defined a colorspace we need to defined parameters all the color
 * hierarchy, because each colorspace is defined relatively to another. Finally,
 * they are all defined directly or indirectly to XYZ.
 * 
 * @example HSL colorspace hierachy is [XYZ, RGB, HSL] and XYZ has no parameter,
 * so you need to set RGBParams and HSLParams.
 */
template<class Color>
struct FullColorParams : public IFullColorParams
{
	typedef FullColorParams<Color> This;
	
	typedef color_dependencies<Color> dependencies;
	typedef typename dependencies::from_root from_root;
	typedef typename dependencies::to_root to_root;
	typedef typename dependencies::size size;
	typedef typename dependencies::color_steps_from_root color_steps_from_root;
	typedef typename dependencies::color_steps_to_root color_steps_to_root;
	
	typedef typename ::boost::mpl::transform<from_root, color_to_colorparams_mplfunc< ::boost::mpl::_1 > >::type full_params; ///< all params from root to Color
	typedef typename ::boost::fusion::result_of::as_vector<full_params>::type full_params_v;
	full_params_v _params;
	
	/**
	 * @brief Create a dynamic view on the current ::boost::mpl::vector values.
	 * @return std::vector of pointers to each color params of the hierachy.
	 */
	std::vector<IColorParams*> getColorspaceVectorView()
	{
		std::vector<IColorParams*> vec;
		vec.reserve( getNbReferences() );
		
//		std::cout << "at_c 0: " << &fusion::at_c<0>(_params) << std::endl;
//		std::cout << "at_c 0 xyzValue: " << fusion::at_c<0>(_params)._xyzValue << std::endl;
		::boost::fusion::for_each( _params, append_abstractptr_from_fusionvector_to_stdvector<IColorParams>( vec ) );
		return vec;
	}
	const std::vector<IColorParams*> getColorspaceVectorView() const { return const_cast<This&>(*this).getColorspaceVectorView(); }
	
	/**
	 * @brief Color hierarchy size.
	 */
	std::size_t getNbReferences() const
	{
		return size::value;
	}
};

/**
 * @brief Mpl functor to convert a list of color types pair into a list of color transforms.
 * @example [HSL>RGB, RGB>XYZ] to [ColorTranform<HSL,RGB>, ColorTransform<RGB,XYZ>]
 */
template< typename ColorPair, bool direction_is_from_root >
struct colorsteps_to_colortransforms_mplfunc
{
	typedef ColorTransformation<
			typename ::boost::mpl::at_c<ColorPair,0>::type, // source color
			typename ::boost::mpl::at_c<ColorPair,1>::type,  // dst color
			direction_is_from_root
		> type;
};

template< typename Color, typename ChannelType >
struct colors_to_pixels_mplfunc
{
	typedef pixel<ChannelType, typename Color::layout> type;
};

/**
 * @brief All transformations from root_colorspace to colorspace (or from colorspace to root_colorspace)
 * 
 * @todo
 * 
 * @example HSL colorspace transformations are:
 * * from_root: [XYZ>RGB, RGB>HSL]
 * * to_root: [HSL>RGB, RGB>XYZ]
 */
template<class Color, typename ChannelType, bool direction_is_from_root>
struct FullColorTransformations
{
	typedef FullColorTransformations<Color, ChannelType, direction_is_from_root> This;
	
	typedef color_dependencies<Color> dependencies;
	typedef typename dependencies::size size;
	
	typedef typename ::boost::mpl::if_c<direction_is_from_root,
			typename dependencies::color_steps_from_root,
			typename dependencies::color_steps_to_root> colorSteps;
	
	typedef typename ::boost::mpl::transform<colorSteps, colorsteps_to_colortransforms_mplfunc< ::boost::mpl::_1, direction_is_from_root > >::type ColorTransformVecT;
	typedef typename ::boost::fusion::result_of::as_vector<ColorTransformVecT>::type ColorTransformVec;
	
	ColorTransformVec _colorTranformVec;

	
	typedef typename ::boost::mpl::if_c<direction_is_from_root,
			typename dependencies::from_root,
			typename dependencies::to_root> colors;
	
	typedef typename ::boost::mpl::transform<colors, colors_to_pixels_mplfunc< ::boost::mpl::_1, ChannelType > >::type PixelVecT;
	typedef typename ::boost::fusion::result_of::as_vector<PixelVecT>::type PixelVec;
	
	PixelVec _pixelVec;
	
	
	/**
	 * @brief Create a dynamic view on the current ::boost::mpl::vector values.
	 * @return std::vector of pointers to each color params of the hierachy.
	 */
	std::vector<IColorTransformation*> getTransformationVectorView()
	{
		std::vector<IColorTransformation*> vec;
		vec.reserve( getNbReferences() );
		
//		std::cout << "at_c 0: " << &fusion::at_c<0>(_colorTranformVec) << std::endl;
//		std::cout << "at_c 0 xyzValue: " << fusion::at_c<0>(_colorTranformVec)._xyzValue << std::endl;
		::boost::fusion::for_each( _colorTranformVec, append_abstractptr_from_fusionvector_to_stdvector<IColorTransformation>( vec ) );
		return vec;
	}
	const std::vector<IColorTransformation*> getTransformationVectorView() const { return const_cast<This&>(*this).getTransformationVectorView(); }
	
	/**
	 * @brief Color hierarchy size.
	 */
	std::size_t getNbReferences() const
	{
		return size::value;
	}
};

/**
 * @brief Find the number of common colorspaces from root color.
 * @param other Another parameter hierachy of values.
 */
std::size_t nbCommonColorspace( const IFullColorParams& a, const IFullColorParams& b )
{
	const std::vector<IColorParams*> aVec = a.getColorspaceVectorView();
	const std::vector<IColorParams*> bVec = b.getColorspaceVectorView();
	const std::size_t maxSize = std::min( aVec.size(), bVec.size() );
	for( std::size_t i = 0; i < maxSize; ++i )
	{
		if( (*aVec[i]) != (*bVec[i]) )
			return i;
	}
	return maxSize;
}

//boost::ptr_vector<IColorTransformation> getColorTransformations( const IFullColorParams& a, const IFullColorParams& b )
//{
//	const std::size_t start = nbCommonColorspace( a, b );
//	boost::ptr_vector<IColorTransformation> transformations;
//	
//	return transformations;
//}


//	template<typename SChannelType, typename DChannelType>
//	void apply( const pixel<SChannelType,Color::layout>& src, pixel<SChannelType,XYZ::layout>& dst ) const
//	{
//	}
//	template<typename SChannelType, typename DChannelType>
//	void apply( const pixel<SChannelType,XYZ::layout>& , pixel<SChannelType,Color::layout>&  ) const
//	{
//	}


struct ApplyColorTransformations
{
	template<typename CT>
	void operator()( CT& colorTransformation ) const
	{
		
	}
};


template<typename Color, typename SChannelType, typename DChannelType>
void color_transformation(
		const FullColorParams<Color>& params,
		const pixel<SChannelType, typename Color::layout>& src,
		pixel<DChannelType,XYZ::layout>& dst )
{
	using namespace terry;
	using namespace terry::numeric;
	
	FullColorTransformations<Color, SChannelType, /*to_root*/false> transformations;
	
	::boost::fusion::for_each( transformations._colorTranformVec, ApplyColorTransformations() );
}


/*
FullColorParams<cmyk> cmykParams;
cmykParams.get<rgb>().colorTemperature = d65;

color_transformation( cmykParams, pixCmyk, adobe_sRgb, pixRgb );
*/

}
}

#endif

