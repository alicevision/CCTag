function [N,s] = normalize_matr_det_1(M)
% Normalise la matrice M en M1 de telle façon que M1 = s*M (s scalaire non nul) et det(M) = 1
   [nr,nc]=size(M);
   if nr == nc,
      d = det(M);
      s = sign(d)/abs(d)^(1/nr);
      N = s * M;
   else
      error('Error in normalize_matr_det_1().');
   end
   