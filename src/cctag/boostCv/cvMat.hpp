#ifndef _CCTAG_CVMAT_HPP
#define	_CCTAG_CVMAT_HPP
#if 0
#ifndef BOOST_UBLAS_SHALLOW_ARRAY_ADAPTOR
#define BOOST_UBLAS_SHALLOW_ARRAY_ADAPTOR
#endif

#include <rom/numerical/algebra/matrix/Matrix.hpp>
#include <rom/numerical/algebra/matrix/io.hpp>
#include <rom/baseType/Point.hpp>

#include <opencv2/legacy/compat.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/core/types_c.h>

#include <boost/numeric/ublas/matrix.hpp>

namespace cctag {
namespace boostCv {

	/**
	 * @brief Retrieve OpenCV defines from template parameters
	 */
	template<class T> struct CvMatGetBaseType { /*CvMatGetType(){ BOOST_STATIC_ASSERT(false); }*/ };
	template<> struct CvMatGetBaseType<double> { static const int value = CV_64F; };
	template<> struct CvMatGetBaseType<float> { static const int value = CV_32F; };
	template<> struct CvMatGetBaseType<unsigned char> { static const int value = CV_8U; };
	template<> struct CvMatGetBaseType<char> { static const int value = CV_8S; };
	template<> struct CvMatGetBaseType<ushort> { static const int value = CV_16U; };
	template<> struct CvMatGetBaseType<short> { static const int value = CV_16S; };

	template< class T, int ND=1 > struct CvMatGetType { static const int value = CV_MAKETYPE( CvMatGetBaseType<T>::value, ND ); };

	/**
	 * @brief Container for OpenCv Matrix Header. It contains no buffer.
	 *
	 * It only owns the matrix structure allocation.
     */
	class CvMatView
	{
	public:
		CvMatView()
		: _mat(NULL)
		{}
		CvMatView( CvMat* const mat )
		: _mat(mat)
		{}
		CvMatView( CvMatView& other )
		{
			_mat = other.release();
		}
		CvMatView( const CvMatView& other )
		{
			_mat = other._mat;
			other._mat = NULL;
		}
		~CvMatView()
		{
			reset();
		}

		CvMatView& operator=( CvMat* mat )
		{
			reset( mat );
			return *this;
		}
		void reset( CvMat* mat = NULL )
		{
			if( _mat )
				cvReleaseMatHeader( &_mat );
			_mat = mat;
		}

		const CvMat* const get() const { return _mat; }
		      CvMat*       get()       { return _mat; }
		const CvMat& getRef() const { return *_mat; }
		      CvMat& getRef()       { return *_mat; }

		CvMat* release() { CvMat* m = _mat; _mat = NULL; return m; }

	private:
		mutable CvMat* _mat;
	};

	/**
	 * @brief Container for OpenCV Matrix.
     */
	class CvMatContainer
	{
	public:
		CvMatContainer()
		: _mat(NULL)
		{}
		CvMatContainer( CvMat* const mat )
		: _mat(mat)
		{}
		CvMatContainer( CvMatContainer& other )
		{
			_mat = other.release();
		}
		CvMatContainer( const CvMatContainer& other )
		{
			_mat = other._mat;
			other._mat = NULL;
		}
		~CvMatContainer()
		{
			reset();
		}

		CvMatContainer& operator=( CvMat* mat )
		{
			reset( mat );
			return *this;
		}
		void reset( CvMat* mat = NULL )
		{
			if( _mat )
				cvReleaseMat( &_mat );
			_mat = mat;
		}

		const CvMat* const get() const { return _mat; }
		CvMat* get() { return _mat; }

		CvMat* release() { CvMat* m = _mat; _mat = NULL; return m; }

	private:
		mutable CvMat* _mat;
	};

//------------------------------------------------------------------------------
// Matrix
//------------------------------------------------------------------------------

	template<typename T, typename StorageType>
	inline CvMatView createCvMatView( boost::numeric::ublas::matrix<T, boost::numeric::ublas::row_major, StorageType> & m )
	{
		CvMat *cvm = cvCreateMatHeader( m.size1(), m.size2(), CvMatGetType<T>::value );
		cvm->data.ptr = reinterpret_cast<unsigned char*>( &m(0,0) );
		return cvm;
	}
	template<typename T, typename StorageType>
	inline const CvMatView createCvMatView( const boost::numeric::ublas::matrix<T, boost::numeric::ublas::row_major, StorageType> & m )
	{
		CvMat* cvm = cvCreateMatHeader( m.size1(), m.size2(), CvMatGetType<T>::value );
		cvm->data.ptr = const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>( &m(0,0) ));
		return cvm;
	}
	template<typename T, typename StorageType>
	inline CvMatView createCvMatTransView( boost::numeric::ublas::matrix<T, boost::numeric::ublas::column_major, StorageType> & m )
	{
		CvMat *cvm = cvCreateMatHeader( m.size2(), m.size1(), CvMatGetType<T>::value );
		cvm->data.ptr = reinterpret_cast<unsigned char*>( &m(0,0) );
		return cvm;
	}
	template<typename T, typename StorageType>
	inline const CvMatView createCvMatTransView( const boost::numeric::ublas::matrix<T, boost::numeric::ublas::column_major, StorageType> & m )
	{
		CvMat* cvm = cvCreateMatHeader( m.size2(), m.size1(), CvMatGetType<T>::value );
		cvm->data.ptr = const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>( &m(0,0) ));
		return cvm;
	}

//------------------------------------------------------------------------------
// Matrix range
//------------------------------------------------------------------------------
	template<typename T, typename StorageType>
	inline CvMatView createCvMatView( boost::numeric::ublas::matrix_range< boost::numeric::ublas::matrix<T, boost::numeric::ublas::row_major, StorageType> > & m )
	{
		CvMat* cvm = cvCreateMatHeader( m.size1(), m.size2(), CvMatGetType<T>::value );
		cvInitMatHeader( cvm, m.size1(), m.size2(), CvMatGetType<T>::value, &m(0,0), m.data().size2() * sizeof(T) );
//		cvm->step = CV_ELEM_SIZE(typename M::value_type) * m.size2() /*m.realsize2()*/;
//		cvm->data.ptr = reinterpret_cast<unsigned char*>( &m(0,0) );
		return cvm;
	}
	template<typename T, typename StorageType>
	inline CvMatView createCvMatTransView( boost::numeric::ublas::matrix_range< boost::numeric::ublas::matrix<T, boost::numeric::ublas::column_major, StorageType> > & m )
	{
		CvMat* cvm = cvCreateMatHeader( m.size2(), m.size1(), CvMatGetType<T>::value );
		cvInitMatHeader( cvm, m.size2(), m.size1(), CvMatGetType<T>::value, &m(0,0), m.data().size1() * sizeof(T) );
//		cvm->step = CV_ELEM_SIZE(typename M::value_type) * m.size2() /*m.realsize2()*/;
//		cvm->data.ptr = reinterpret_cast<unsigned char*>( &m(0,0) );
		return cvm;
	}
	template<typename T, typename StorageType>
	inline CvMatView createCvMatView( const boost::numeric::ublas::matrix_range< boost::numeric::ublas::matrix<T, boost::numeric::ublas::row_major, StorageType> > & m )
	{
		CvMat* cvm = cvCreateMatHeader( m.size1(), m.size2(), CvMatGetType<T>::value );
		cvInitMatHeader( cvm, m.size1(), m.size2(), CvMatGetType<T>::value, const_cast<T*>(&m(0,0)), m.data().size2() * sizeof(T) );
//		cvm->step = CV_ELEM_SIZE(typename M::value_type) * m.size2() /*m.realsize2()*/;
//		cvm->data.ptr = reinterpret_cast<unsigned char*>( &m(0,0) );
		return cvm;
	}
	template<typename T, typename StorageType>
	inline CvMatView createCvMatTransView( const boost::numeric::ublas::matrix_range< boost::numeric::ublas::matrix<T, boost::numeric::ublas::column_major, StorageType> > & m )
	{
		CvMat* cvm = cvCreateMatHeader( m.size2(), m.size1(), CvMatGetType<T>::value );
		cvInitMatHeader( cvm, m.size2(), m.size1(), CvMatGetType<T>::value, const_cast<T*>(&m(0,0)), m.data().size1() * sizeof(T) );
//		cvm->step = CV_ELEM_SIZE(typename M::value_type) * m.size2() /*m.realsize2()*/;
//		cvm->data.ptr = reinterpret_cast<unsigned char*>( &m(0,0) );
		return cvm;
	}

	template<typename T, typename StorageType>
	inline CvMatView createCvMatTransView( const boost::numeric::ublas::matrix_range< boost::numeric::ublas::matrix_range< boost::numeric::ublas::matrix<T, boost::numeric::ublas::column_major, StorageType> > > & m )
	{
		CvMat* cvm = cvCreateMatHeader( m.size2(), m.size1(), CvMatGetType<T>::value );
		cvInitMatHeader( cvm, m.size2(), m.size1(), CvMatGetType<T>::value, const_cast<T*>(&m(0,0)), m.data().data().size1() * sizeof(T) );
//		cvm->step = CV_ELEM_SIZE(typename M::value_type) * m.size2() /*m.realsize2()*/;
//		cvm->data.ptr = reinterpret_cast<unsigned char*>( &m(0,0) );
		return cvm;
	}

	template<typename T, typename StorageType>
	inline CvMatView createCvMatView( boost::numeric::ublas::matrix_range< const boost::numeric::ublas::matrix<T, boost::numeric::ublas::row_major, StorageType> > & m )
	{
		CvMat* cvm = cvCreateMatHeader( m.size1(), m.size2(), CvMatGetType<T>::value );
		cvInitMatHeader( cvm, m.size1(), m.size2(), CvMatGetType<T>::value, const_cast<T*>(&m(0,0)), m.data().size2() * sizeof(T) );
//		cvm->step = CV_ELEM_SIZE(typename M::value_type) * m.size2() /*m.realsize2()*/;
//		cvm->data.ptr = reinterpret_cast<unsigned char*>( &m(0,0) );
		return cvm;
	}
	template<typename T, typename StorageType>
	inline CvMatView createCvMatTransView( boost::numeric::ublas::matrix_range< const boost::numeric::ublas::matrix<T, boost::numeric::ublas::column_major, StorageType> > & m )
	{
		CvMat* cvm = cvCreateMatHeader( m.size2(), m.size1(), CvMatGetType<T>::value );
		cvInitMatHeader( cvm, m.size2(), m.size1(), CvMatGetType<T>::value, const_cast<T*>(&m(0,0)), m.data().size1() * sizeof(T) );
//		cvm->step = CV_ELEM_SIZE(typename M::value_type) * m.size2() /*m.realsize2()*/;
//		cvm->data.ptr = reinterpret_cast<unsigned char*>( &m(0,0) );
		return cvm;
	}
	template<typename T, typename StorageType>
	inline CvMatView createCvMatView( const boost::numeric::ublas::matrix_range< const boost::numeric::ublas::matrix<T, boost::numeric::ublas::row_major, StorageType> > & m )
	{
		CvMat* cvm = cvCreateMatHeader( m.size1(), m.size2(), CvMatGetType<T>::value );
		cvInitMatHeader( cvm, m.size1(), m.size2(), CvMatGetType<T>::value, const_cast<T*>(&m(0,0)), m.data().size2() * sizeof(T) );
//		cvm->step = CV_ELEM_SIZE(typename M::value_type) * m.size2() /*m.realsize2()*/;
//		cvm->data.ptr = reinterpret_cast<unsigned char*>( &m(0,0) );
		return cvm;
	}
	template<typename T, typename StorageType>
	inline CvMatView createCvMatTransView( const boost::numeric::ublas::matrix_range< const boost::numeric::ublas::matrix<T, boost::numeric::ublas::column_major, StorageType> > & m )
	{
		CvMat* cvm = cvCreateMatHeader( m.size2(), m.size1(), CvMatGetType<T>::value );
		cvInitMatHeader( cvm, m.size2(), m.size1(), CvMatGetType<T>::value, const_cast<T*>(&m(0,0)), m.data().size1() * sizeof(T) );
//		cvm->step = CV_ELEM_SIZE(typename M::value_type) * m.size2() /*m.realsize2()*/;
//		cvm->data.ptr = reinterpret_cast<unsigned char*>( &m(0,0) );
		return cvm;
	}

	template<typename T, typename StorageType>
	inline CvMatView createCvMatTransView( const boost::numeric::ublas::matrix_range< const boost::numeric::ublas::matrix_range< boost::numeric::ublas::matrix<T, boost::numeric::ublas::column_major, StorageType> > > & m )
	{
		CvMat* cvm = cvCreateMatHeader( m.size2(), m.size1(), CvMatGetType<T>::value );
		cvInitMatHeader( cvm, m.size2(), m.size1(), CvMatGetType<T>::value, const_cast<T*>(&m(0,0)), m.data().data().size1() * sizeof(T) );
//		cvm->step = CV_ELEM_SIZE(typename M::value_type) * m.size2() /*m.realsize2()*/;
//		cvm->data.ptr = reinterpret_cast<unsigned char*>( &m(0,0) );
		return cvm;
	}

	//------------------------------------------------------------------------------
// Bounded Matrix
//------------------------------------------------------------------------------

	template<typename T, std::size_t S1, std::size_t S2>
	inline CvMatView createCvMatView( boost::numeric::ublas::bounded_matrix<T, S1, S2, boost::numeric::ublas::row_major> & m )
	{
		CvMat *cvm = cvCreateMatHeader( m.size1(), m.size2(), CvMatGetType<T>::value );
		cvm->data.ptr = reinterpret_cast<unsigned char*>( &m(0,0) );
		return cvm;
	}
	template<typename T, std::size_t S1, std::size_t S2>
	inline const CvMatView createCvMatView( const boost::numeric::ublas::bounded_matrix<T, S1, S2, boost::numeric::ublas::row_major> & m )
	{
		CvMat *cvm = cvCreateMatHeader( m.size1(), m.size2(), CvMatGetType<T>::value );
		cvm->data.ptr = const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>( &m(0,0) ));
		return cvm;
	}


	template<typename T, std::size_t S1, std::size_t S2>
	inline CvMatView createCvMatTransView( boost::numeric::ublas::bounded_matrix<T, S1, S2, boost::numeric::ublas::column_major> & m )
	{
		CvMat *cvm = cvCreateMatHeader( m.size2(), m.size1(), CvMatGetType<T>::value );
		cvm->data.ptr = reinterpret_cast<unsigned char*>( &m(0,0) );
		return cvm;
	}
	template<typename T, std::size_t S1, std::size_t S2>
	inline const CvMatView createCvMatTransView( const boost::numeric::ublas::bounded_matrix<T, S1, S2, boost::numeric::ublas::column_major> & m )
	{
		CvMat *cvm = cvCreateMatHeader( m.size2(), m.size1(), CvMatGetType<T>::value );
		cvm->data.ptr = const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>( &m(0,0) ));
		return cvm;
	}

//------------------------------------------------------------------------------
// Vector
//------------------------------------------------------------------------------
	template<typename T, typename StorageType>
	inline CvMatView createCvMatView( boost::numeric::ublas::vector<T, StorageType> & m )
	{
		CvMat *cvm = cvCreateMatHeader( 1, m.size(), CvMatGetType<T>::value );
		cvm->data.ptr = reinterpret_cast<unsigned char*>( &m(0) );
		return cvm;
	}
	template<typename T, typename StorageType>
	inline const CvMatView createCvMatView( const boost::numeric::ublas::vector<T, StorageType> & m )
	{
		CvMat *cvm = cvCreateMatHeader( 1, m.size(), CvMatGetType<T>::value );
		cvm->data.ptr = const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>( &m(0) ));
		return cvm;
	}

	template<typename T, typename StorageType>
	inline CvMatView createCvMatTransView( boost::numeric::ublas::vector<T, StorageType> & m )
	{
		CvMat *cvm = cvCreateMatHeader( m.size(), 1, CvMatGetType<T>::value );
		cvm->data.ptr = reinterpret_cast<unsigned char*>( &m(0) );
		return cvm;
	}
	template<typename T, typename StorageType>
	inline const CvMatView createCvMatTransView( const boost::numeric::ublas::vector<T, StorageType> & m )
	{
		CvMat *cvm = cvCreateMatHeader( m.size(), 1, CvMatGetType<T>::value );
		cvm->data.ptr = const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>( &m(0) ));
		return cvm;
	}

//------------------------------------------------------------------------------
// Bounded Vector
//------------------------------------------------------------------------------
	template<typename T, std::size_t S>
	inline CvMatView createCvMatView( boost::numeric::ublas::bounded_vector<T, S> & m )
	{
		CvMat *cvm = cvCreateMatHeader( 1, m.size(), CvMatGetType<T>::value );
		cvm->data.ptr = const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>( &m(0) ));
		return cvm;
	}
	template<typename T, std::size_t S>
	inline const CvMatView createCvMatView( const boost::numeric::ublas::bounded_vector<T, S> & m )
	{
		CvMat *cvm = cvCreateMatHeader( 1, m.size(), CvMatGetType<T>::value );
		cvm->data.ptr = const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>( &m(0) ));
		return cvm;
	}

	template<typename T, std::size_t S>
	inline CvMatView createCvMatTransView( boost::numeric::ublas::bounded_vector<T, S> & m )
	{
		CvMat *cvm = cvCreateMatHeader( m.size(), 1, CvMatGetType<T>::value );
		cvm->data.ptr = const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>( &m(0) ));
		return cvm;
	}
	template<typename T, std::size_t S>
	inline const CvMatView createCvMatTransView( const boost::numeric::ublas::bounded_vector<T, S> & m )
	{
		CvMat *cvm = cvCreateMatHeader( m.size(), 1, CvMatGetType<T>::value );
		cvm->data.ptr = const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>( &m(0) ));
		return cvm;
	}

//------------------------------------------------------------------------------
// Utility functions for CvMatView and CvMatContainer creation.
//------------------------------------------------------------------------------

	/**
	 * @todo specialization for std::vector< Point2d<Eigen::Vector3f> >
	 */
	template<typename T>
	inline CvMatView createCvMatView( std::vector<T> & m )
	{
		CvMat *cvm = cvCreateMatHeader( 1, m.size(), CvMatGetType<T>::value );
		cvm->data.ptr = reinterpret_cast<unsigned char*>( &m[0] );
		return cvm;
	}

	template<typename T>
	inline CvMatView createCvMatView( boost::numeric::ublas::vector<T> & m )
	{
		CvMat *cvm = cvCreateMatHeader( 1, m.size(), CvMatGetType<T>::value );
		cvm->data.ptr = reinterpret_cast<unsigned char*>( &m[0] );
		return cvm;
	}

	template<typename T>
	inline CvMatView createCvMatView( const boost::numeric::ublas::vector_range< boost::numeric::ublas::vector<T> > & m )
	{
		CvMat *cvm = cvCreateMatHeader( 1, m.size(), CvMatGetType<T>::value );
		cvm->data.ptr = const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>( &m[0] ) );
		return cvm;
	}

	template<typename T>
	inline CvMatView createCvMatView( boost::numeric::ublas::vector_range< boost::numeric::ublas::vector<T> > & m )
	{
		CvMat *cvm = cvCreateMatHeader( 1, m.size(), CvMatGetType<T>::value );
		cvm->data.ptr = reinterpret_cast<unsigned char*>( &m[0] );
		return cvm;
	}

	/// @brief Not allowed views
	/// @{
	template<typename T>
	inline CvMatView createCvMatView( std::vector<Point2dN<T> > & m );
	template<typename T>
	inline const CvMatView createCvMatView( const std::vector<Point2dN<T> > & m );
	/// @}

	template<typename T, std::size_t S1, std::size_t S2>
	inline CvMatContainer createCvMatContainer( typename cctag::numerical::BoundedMatrix<T,S1,S2>::Type & m )
	{
		CvMat *cvm = cvCloneMat( createCvMatView(m).get() );
		return cvm;
	}

}
}


inline std::ostream& operator<<( std::ostream& os, const cctag::boostCv::CvMatView & matrix )
{
	os << *(matrix.get());
	return os;
}

inline std::ostream& operator<<( std::ostream& os, const cctag::boostCv::CvMatContainer & matrix )
{
	os << *matrix.get();
	return os;
}

#endif
#endif
