function getTw()

ratio = 0.4;
outImsize = [640,400]*ratio;
f_vir = 200*ratio;

K = [f_vir 0  outImsize(2)/2;
         0 f_vir  outImsize(1)/2;
         0 0 1];
%Rref =[1 0 0; 0 0 1; 0 -1 0]; -> 
Rref =[0 0 1; 0 -1 0; 1 0 0];
viewImgsize = [256,160];

cx = K(1,3); cy = K(2,3);  
fx = K(1,1); fy = K(2,2); 

[xi,yi] = meshgrid(1:viewImgsize(2), 1:viewImgsize(1));   
Ax = (xi-cx)./fx; 
By = (yi-cy)./fy; 
T = [Ax(:)';By(:)'; ones(1,prod(viewImgsize))];
Twall = zeros(viewImgsize(1),viewImgsize(2)*4,3);
for virtual_cam_i = 1:4
     R = getRotationMatrix('y',-(virtual_cam_i-1)*pi/2);
     vir_cam_extrinsic{virtual_cam_i} = [R(1:3,1:3)*Rref];
     Tw = vir_cam_extrinsic{virtual_cam_i}(1:3,1:3)*T;
     Twall(:,1+viewImgsize(2)*(virtual_cam_i-1):viewImgsize(2)*virtual_cam_i,:) = reshape(Tw',[viewImgsize,3]);
end

Twall = -Twall;
Twall = permute(Twall,[2,1,3]);
hdf5write('/n/fs/modelnet/roomEncoder/context-encoder/Tw.h5','Tw',Twall);
%{
filename = '/n/fs/modelnet/roomEncoder/context-encoder/checkpoints/suncg_olw1_8roomest_xyz_twoview_rgbpsn/debug.h5';
normal_est = hdf5read(filename,'normal_est');
normal = hdf5read(filename,'normal');
pm1 = hdf5read(filename,'pm1');

nm = permute(normal(:,:,:,1),[2,1,3]);
pm = permute(pm1(:,:,:,1),[2,1,3]);




filename = '/n/fs/modelnet/roomEncoder/context-encoder/checkpoints/suncg_olw1_8roomest_xyz_twoview_rgbpsn/debugXYZ_gt.h5';
XYZ_gt = hdf5read(filename,'XYZ_gt');
XYZ_batch = permute(XYZ_gt(:,:,:,1),[2,1,3]);

data_input = hdf5read(filename,'result');
norm_real =  data_input(:,:,6:8,batchId);
depth_input = permute((data_input(:,:,4,batchId)+1)*0.25*65535*0.25*0.001,[2,1]);

XYZ =  hdf5read(filename,'XYZ');
XYZ_batch = permute(XYZ(:,:,:,batchId),[2,1,3]);
pmap = permute(hdf5read(filename,'pm1'),[2,1,3,4]);


Points_cam = (depth_input./sum(Twall.*norm_real,3));
Points = Twall.*repmat(Points_cam,[1,1,3]);
%}
end