function [rgb,points3d] = panoimgPlane2point(imageview_pano,pmapview_pano,normal_pano,vir_cam_center,pt,K_vir) 
%         outImsize = [640,400]*ratio;
%         f_vir = 200*ratio;
%         
%         K_vir = [f_vir 0  outImsize(2)/2;
%                  0 f_vir  outImsize(1)/2;
%                  0 0 1];
      %Rref =[1 0 0; 0 0 1; 0 -1 0]; -> 
      Rref =[0 0 1; 0 -1 0; 1 0 0];
      for virtual_cam_i = 4:-1:1
         %R = getRotationMatrix('z',-(virtual_cam_i-1)*pi/2);
         R = getRotationMatrix('y',-(virtual_cam_i-1)*pi/2);
         vir_cam_extrinsic{virtual_cam_i} = [R(1:3,1:3)*Rref,vir_cam_center(1:3)];
      end
      gap = size(imageview_pano,2)/4;
      for virtual_cam_i = 1:4      
          imageview = imageview_pano(:,1+gap*(virtual_cam_i-1):gap*virtual_cam_i,:);
          pmap = pmapview_pano(:,1+gap*(virtual_cam_i-1):gap*virtual_cam_i,:);
          normal = normal_pano(:,1+gap*(virtual_cam_i-1):gap*virtual_cam_i,:);
          rgb_v = reshape(imageview, [], 3);

          [points3d_v,points3dw_v] = pmap2XYZ(pmap,normal,K_vir,vir_cam_extrinsic{virtual_cam_i},size(pmap));
          if pt>0
             inx2remove = removeNosiePoints(points3dw_v,pt);
             points3dw_v(:,find(inx2remove(:))) = NaN;
          end

          rgb{virtual_cam_i} = rgb_v';
          points3dw_v = bsxfun(@plus,points3dw_v,vir_cam_center);
          points3d{virtual_cam_i} = points3dw_v;

      end