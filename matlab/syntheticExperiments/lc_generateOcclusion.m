function [I,rRes] = lc_generateOcclusion(pattern,pOcclusion, radiusPixel)

I = pattern.I;

[height, width] = size(I);

if ( width ~= height )
    warning('Argument passed in generate occlusion in not a square image');
end

p  = 0;
i = 1;

if strcmp(pattern.typePrimitive, 'circular')
    totalArea = pi*radiusPixel*radiusPixel;
    C1 = param2ellipse([width/2 height/2 radiusPixel radiusPixel 0]);
    % Loop on ordonates in the image
    while p < pOcclusion*totalArea
        
        % Compute round intersections with line y = i and C1 and C2
        x1 = round(intersectEllipseLine(C1,i));
        
        % Treat all the cases, respectively (2,2) intersections with (C1,C2),
        % (2,1) intersections with (C1,C2),...
        if (length(x1) == 1)
            I(i,x1) = 255;
            p = p+1;
        elseif (length(x1) == 2)
            nb = abs(x1(1)-x1(2));
            p = p+nb;
            I(i,x1(1):x1(2)) = 255;
        end
        i = i+1;
    end
elseif strcmp(pattern.typePrimitive, 'square')
    totalArea = width*height;
    % Loop on ordonates in the image
    while p < pOcclusion*totalArea
        p = p+size(I,2);
        I(i,:) = 255;
        i = i+1;
    end
    %I = imrotate(I,rand*90,'nearest','crop');
end
rRes = i-1;
	