function blendIm = linearBlending(im1,im2,mask1,mask2)
    %% mask is the weight of color 
    mask1 = logical(mask1);
    mask2 = logical(mask2);
    mask1only = mask1 & ~mask2;
    mask2only = ~mask1 & mask2;
    intersection = mask1 & mask2;
    %  bwdist(mask1only) all pixels distance to the mask's distance
    closeTo1 = bwdist(mask1only) < bwdist(mask2only);
    
    mask1 = mask1only | (intersection & closeTo1);% mask 1 only  or the regoin in iversection close to 1 
    mask2 = mask2only | (intersection & ~closeTo1);
    
    %% blend
    sd = 10;
    blendIm = zeros(size(im1));
    fil = fspecial('gaussian', 1+3*sd, sd);
    mask1Curr = imfilter(double(mask1),fil); 
    mask1Curr(mask1only) = 1;% mask1 only regoin not change black regoin not get into 1
    mask1Curr(mask2only) = 0;% im1 dn't want to get into 2
    
    mask2Curr = imfilter(double(mask2),fil); 
    mask2Curr(mask2only) = 1; 
    mask2Curr(mask1only) = 0;
    
    % make sure weight 1 + weight 2 = 1
    tmpR = 0 < mask1Curr + mask2Curr & mask1Curr + mask2Curr < 1; % same empty pixel get in!  
    tmpS = mask1Curr + mask2Curr;
    mask1Curr(tmpR) = mask1Curr(tmpR) ./ tmpS(tmpR);
    mask2Curr(tmpR) = mask2Curr(tmpR) ./ tmpS(tmpR);
    for i = 1:size(im1,3)
        blendIm(:,:,i) = blendIm(:,:,i) + mask1Curr.*im1(:,:,i) + mask2Curr.*im2(:,:,i);
    end
    
end