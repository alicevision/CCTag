#ifndef _LAPACK_ADD_H
#define	_LAPACK_ADD_H

#include <boost/numeric/bindings/traits/type.h>

#ifndef BOOST_NUMERIC_BINDINGS_USE_CLAPACK
#  include <boost/numeric/bindings/traits/fortran.h>
#else
#  define FORTRAN_ID( id ) id##_
#endif

/********************************************/
/* generalized eigenvalue/eigenvector */

#define LAPACK_DGGEV FORTRAN_ID( dggev )

/********************************************/
/* norm computation */

#define LAPACK_DLANGE FORTRAN_ID( dlange )

#ifndef BOOST_NUMERIC_BINDINGS_NO_STRUCTURE_CHECK
#  define BOOST_NUMERIC_BINDINGS_FORTRAN
#endif

#ifdef	__cplusplus
extern "C" {
#endif

/********************************************/
/* generalized eigenvalue/eigenvector */


	void LAPACK_DGGEV(char const* jobz, char *jobvr, int const* n,
                     double* a, int const* lda, double* b, int const* ldb,
                     double *alphar, double *alphai,
					 double *beta, double *vl, int *ldvl, double *vr,
					 int *ldvr, double *work, int *lwork, int *info);

/********************************************/
/* norm computation */

	double LAPACK_DLANGE( char *norm, int *m, int *n, double *a, int *lda, double *work );

#ifdef	__cplusplus
}
#endif

#endif	/* _LAPACK_ADD_H */

