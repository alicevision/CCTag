#ifndef _CCTAG_HOMOGRAPHY_HPP_
#define _CCTAG_HOMOGRAPHY_HPP_

#include <opencv/cv.h>

namespace cctag {
namespace vision {
namespace viewGeometry {

/// @todo: move ellipse_to_params and param_to_ellipse into Ellipse.hpp

//transforme la matrice d'une ellipse en ses paramètres correspondants
void ellipse_to_params( CvMat * E, double param[5] );

//transforme les paramètres d'une ellipse en sa matrice correspondante
void param_to_ellipse( const double x, const double y, const double largeur, const double hauteur, const double angle, CvMat* Ellipse );

//retrouve l'homographie faisant passer le plan de la caméra au plan de la mire
//E0 et E1 correspondent aux matrices des deux ellipses, images de cercles concentriques
//E0 correspond à la matrice de l'ellipse la plus grande
CvMat retrouve_homographie( const CvMat* E0, const CvMat* E1, const int largeur, const int hauteur );

}
}
}

#endif
