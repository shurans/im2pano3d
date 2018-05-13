function boundaryOut = mask2ploy(mask,TOL)
mask = virtualDep{virtual_cam_i}>0; 
boundaries = bwboundaries(mask);
boundaryOut = {};
for i=1:length(boundaries)
    boundary = boundaries{i};
    if size(boundary,1)>10
        hold on
        plot(boundary(:,2),boundary(:,1),'r.');
        [xo,yo]= reducem(boundary(:,2),boundary(:,1),TOL);
        if ~isempty(xo)
            boundaryOut{end+1} = [xo,yo];
        end
        
    end
end
