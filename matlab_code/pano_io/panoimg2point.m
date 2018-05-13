function [rgb,points3d] = panoimg2point(imageview_pano,depthview_pano, vir_cam_center,K_vir)

        %{
        outImsize = [640,400]*ratio;
        f_vir = 200*ratio;
        
        K_vir = [f_vir 0  outImsize(2)/2;
                 0 f_vir  outImsize(1)/2;
                 0 0 1];
        %}
        
        Rref =[0 0 1; 0 -1 0; 1 0 0];
        for virtual_cam_i = 1:4
            R = getRotationMatrix('y',-(virtual_cam_i-1)*pi/2);
            vir_cam_extrinsic{virtual_cam_i} = [R(1:3,1:3)*Rref,vir_cam_center(1:3)];
        end
        

        gap = size(depthview_pano,2)/4;
           
        for virtual_cam_i = 1:4
           imageview = imageview_pano(:,1+gap*(virtual_cam_i-1):gap*virtual_cam_i,:);
           depthview = depthview_pano(:,1+gap*(virtual_cam_i-1):gap*virtual_cam_i,:);
           [rgb_v,points3d_v]=read_3d_pts_general(depthview,K_vir,size(depthview),imageview);
           rgb{virtual_cam_i} = rgb_v';
           points3d{virtual_cam_i} = transformPointCloud(points3d_v',vir_cam_extrinsic{virtual_cam_i});
           %vis_point_cloud(points3d_t',double(rgb)/255); hold on; pause;
        end
           
end