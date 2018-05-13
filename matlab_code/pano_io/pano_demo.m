filename = './panodata/d65b6505904448d1940e679c9a098047';
%% Demo 1: from depth map to point cloud
% read in depth convert it to meters
depth = double(imread([filename '_d_r0.4.png']))*0.25/1000;
% read in normal map convert it to [-1,1]
normal = double(imread([filename '_n_r0.4.png']))/255*2-1;
% read in color image
color = double(imread([filename '_i_r0.4.jpg']))/255;
% convert rgbd panorama to color point cloud
view_width = 160;
view_height = 256;
f_vir = 80;
K_vir = [f_vir 0  view_width/2;
         0 f_vir view_height/2;
         0 0 1];
[rgb,points3d] = panoimg2point(color,depth, [0,0,0]',K_vir);
figure(1); clf;pcshow(pointCloud(cell2mat(points3d)','Color', cell2mat(rgb)'));


     
%% From depth map and normal map to plane equation 
% camera pose for the skybox 
cameraPose = [0 0 0 1 0 0 0 1 0
              0 0 0 0 0 1 0 1 0
              0 0 0 -1 0 0 0 1 0
              0 0 0 0 0 -1 0 1 0];
      
pmap = zeros(view_height, 4*view_width,1);
for view_idx = 1:4
    d_v = depth(:,(view_idx-1)*view_width+1:view_idx*view_width,:);
    n_v = normal(:,(view_idx-1)*view_width+1:view_idx*view_width,:);
    n_v = reshape(n_v,[],3);
    
    pmap(:,(view_idx-1)*view_width+1:view_idx*view_width,:) = ...
        depth2pmap(d_v, n_v, K_vir,cameraPose(view_idx,:));
end 

figure(2), imagesc(pmap); axis equal;

%% From plane distance and normal to point cloud
[rgb,points3d] = panoimgPlane2point(color,pmap,normal,[0,0,0]',3,K_vir);

figure(3); clf;pcshow(pointCloud(cell2mat(points3d)','Color', cell2mat(rgb)'));
