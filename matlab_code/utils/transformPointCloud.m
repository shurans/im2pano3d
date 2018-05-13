function XYZtransform = transformPointCloud(XYZ,Rt,scale)
if ~exist('scale','var')
    scale =1;
end
    XYZtransform = Rt(1:3,1:3) * scale*XYZ + repmat(Rt(1:3,4),1,size(XYZ,2));
end