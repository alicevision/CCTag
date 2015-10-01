% Plot a the 3D coordinate systeme

U = [1;0;0];
V = [0;1;0];
W = [0;0;1];

q = quiver3(0,0,0,U(1),U(2),U(3));
set(q,'Color','r','LineWidth',4);
q = quiver3(0,0,0,V(1),V(2),V(3));
set(q,'Color','g','LineWidth',4);
q = quiver3(0,0,0,W(1),W(2),W(3));
set(q,'Color','b','LineWidth',4);