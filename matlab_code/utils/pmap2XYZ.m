function [points3d,points3dw] = pmap2XYZ(pmap,normal,K,cam2World,depthInpaintsize)

        cx = K(1,3); cy = K(2,3);  
        fx = K(1,1); fy = K(2,2); 
        
        [xi,yi] = meshgrid(1:depthInpaintsize(2), 1:depthInpaintsize(1));   
        Ax = (xi-cx)./fx; 
        By = (yi-cy)./fy; 

        T = [Ax(:)';By(:)'; ones(1,prod(depthInpaintsize))];
        Tw = cam2World(1:3,1:3)*T;
        normal = reshape(normal,[],3)';

        zdotn = -1*sum(Tw.*normal,1);
        
        z3 = pmap(:)./zdotn(:);
        x3 = Ax(:).*z3;
        y3 = By(:).*z3;
        points3d = [x3(:)'; y3(:)'; z3(:)'];
        
        points3dw = Tw.*repmat(z3',[3,1]);
end