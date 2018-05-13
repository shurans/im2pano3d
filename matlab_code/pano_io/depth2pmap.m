function pmap = depth2pmap(depthRaw,normal, K_vir,cameraPose)
        [~,points3d]=read_3d_pts_general(depthRaw,K_vir,size(depthRaw),[]);
        extCamera2World = camPose2Extrinsics(cameraPose);
        extCamera2World(:,4) = 0; % center around camera 
        points3dW = transformPointCloud(points3d',extCamera2World)';
        pmap = reshape(-1*sum(points3dW.*normal,2),size(depthRaw));
end
