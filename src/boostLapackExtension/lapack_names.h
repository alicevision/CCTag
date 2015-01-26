/* 
 * File:   lapack_names.h
 * Author: edubois
 *
 * Created on December 6, 2010, 2:30 PM
 */

#ifndef _LAPACK_NAMES_H
#define	_LAPACK_NAMES_H

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

#endif	/* _LAPACK_NAMES_H */

