function N=normalize(U)
% ----------------------------------------------------------------------------
  d = size(U,1);
  j=find(U(d,:)==0);
  if ~isempty(j), disp('normalize() ***  DIVISION PAR ZÉRO! [ARRÊT]'); end;
  N = augment(U(1:d-1,:)./repmat(U(d,:),d-1,1));
