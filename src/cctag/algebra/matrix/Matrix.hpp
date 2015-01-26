#ifndef _ROM_NUMERICAL_MATRIX_HPP_
#define	_ROM_NUMERICAL_MATRIX_HPP_

#include <cctag/progBase/Allocator.hpp>

#include <boost/numeric/ublas/storage.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

namespace rom {
namespace numerical {

typedef boost::numeric::ublas::row_major DefaultStorageOrder;


/*
//-----------------------------
// C++-1x solution
template<typename BaseType>
typedef boost::numeric::ublas::matrix<BaseType, StorageOrder, boost::numeric::ublas::unbounded_array<BaseType, rom::AlignedAllocator<BaseType> > > Matrix;

template<typename T>
void f( Matrix<T> & m );

Matrixd m;
f(m);

//-----------------------------
// struct to simulate a template alias
// but doesn't allows automatic template
template<typename T>
void f( typename Matrix<T>::Type m, typename BoundedMatrix<T, 3,4>::Type& mm );

Matrixd m;
f<double>(m);

//-----------------------------
// define solution
// not really clear
template<typename T>
void f( ROM_Matrix(T) & m, ROM_BoundedMatrix(T, 3, 4) & mm );

Matrixd m;
f(m);

//-----------------------------
*/
template<typename T, typename SO = DefaultStorageOrder>
struct Matrix
{
	typedef T BaseType;
	typedef boost::numeric::ublas::unbounded_array<BaseType, rom::AlignedAllocator<BaseType> > StorageType;
	typedef boost::numeric::ublas::matrix<BaseType, SO, StorageType> Type;

	typedef boost::numeric::ublas::shallow_array_adaptor<BaseType>        ShallowStorageType;
	typedef boost::numeric::ublas::matrix<BaseType, SO, ShallowStorageType>   ShallowType;

private:
	Matrix(){}
};

template<class T, std::size_t M, std::size_t N, typename SO = DefaultStorageOrder>
struct BoundedMatrix
{
	typedef T BaseType;
//	typedef boost::numeric::ublas::bounded_array<BaseType, M*N, rom::AlignedAllocator<BaseType> > StorageType;
	typedef boost::numeric::ublas::bounded_matrix<BaseType, M, N, SO> Type;

private:
	BoundedMatrix(){}
};

template<typename T>
struct Vector
{
	typedef T BaseType;
	typedef boost::numeric::ublas::unbounded_array<BaseType, rom::AlignedAllocator<BaseType> > StorageType;
	typedef boost::numeric::ublas::vector<BaseType, StorageType> Type;

	typedef boost::numeric::ublas::shallow_array_adaptor<BaseType>        ShallowStorageType;
	typedef boost::numeric::ublas::vector<BaseType, ShallowStorageType>   ShallowType;

private:
	Vector(){}
};

template<typename T, std::size_t S>
struct BoundedVector
{
	typedef T BaseType;
//	typedef boost::numeric::ublas::bounded_array<BaseType, rom::AlignedAllocator<BaseType> > StorageType;
	typedef boost::numeric::ublas::bounded_vector<BaseType, S> Type;

//	typedef boost::numeric::ublas::shallow_array_adaptor<BaseType>              ShallowStorageType;
//	typedef boost::numeric::ublas::bounded_vector<BaseType, ShallowStorageType> ShallowType;

private:
	BoundedVector(){}
};





#define ROM_DEFINE_NUMERICAL_MATRIX_TYPES( TYPE, ORDER, ORDERSTR, POST ) \
	\
	typedef Matrix<TYPE,ORDER>::Type Matrix##ORDERSTR##POST; \
	typedef Matrix<TYPE,ORDER>::ShallowType MatrixView##ORDERSTR##POST; \
	typedef Matrix<TYPE,ORDER>::ShallowStorageType MatrixViewInit##ORDERSTR##POST; \
	\
	typedef BoundedMatrix<TYPE, 2, 2, ORDER>::Type BoundedMatrix2x2##ORDERSTR##POST; \
	typedef BoundedMatrix<TYPE, 2, 3, ORDER>::Type BoundedMatrix2x3##ORDERSTR##POST; \
	typedef BoundedMatrix<TYPE, 2, 4, ORDER>::Type BoundedMatrix2x4##ORDERSTR##POST; \
	\
	typedef BoundedMatrix<TYPE, 3, 2, ORDER>::Type BoundedMatrix3x2##ORDERSTR##POST; \
	typedef BoundedMatrix<TYPE, 3, 3, ORDER>::Type BoundedMatrix3x3##ORDERSTR##POST; \
	typedef BoundedMatrix<TYPE, 3, 4, ORDER>::Type BoundedMatrix3x4##ORDERSTR##POST; \
	\
	typedef BoundedMatrix<TYPE, 4, 2, ORDER>::Type BoundedMatrix4x2##ORDERSTR##POST; \
	typedef BoundedMatrix<TYPE, 4, 3, ORDER>::Type BoundedMatrix4x3##ORDERSTR##POST; \
	typedef BoundedMatrix<TYPE, 4, 4, ORDER>::Type BoundedMatrix4x4##ORDERSTR##POST; \
//

#define ROM_DEFINE_NUMERICAL_TYPES( TYPE, POST ) \
	\
	ROM_DEFINE_NUMERICAL_MATRIX_TYPES( TYPE, boost::numeric::ublas::row_major, , POST ) \
	ROM_DEFINE_NUMERICAL_MATRIX_TYPES( TYPE, boost::numeric::ublas::column_major, C, POST ) \
	ROM_DEFINE_NUMERICAL_MATRIX_TYPES( TYPE, boost::numeric::ublas::row_major, R, POST ) \
	\
	typedef Vector<TYPE>::Type Vector##POST; \
	typedef Vector<TYPE>::ShallowType VectorView##POST; \
	\
	typedef boost::numeric::ublas::shallow_array_adaptor<TYPE> ArrayViewInit##POST; \
	\
	typedef BoundedVector<TYPE, 2>::Type BoundedVector2##POST; \
	typedef BoundedVector<TYPE, 3>::Type BoundedVector3##POST; \
	typedef BoundedVector<TYPE, 4>::Type BoundedVector4##POST; \
//


ROM_DEFINE_NUMERICAL_TYPES( double, d )
ROM_DEFINE_NUMERICAL_TYPES( float, f )
//ROM_DEFINE_NUMERICAL_TYPES( int, i )
//ROM_DEFINE_NUMERICAL_TYPES( char, c )


}
}

#endif

