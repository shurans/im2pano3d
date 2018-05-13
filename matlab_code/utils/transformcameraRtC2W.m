function cameraRtC2Wout = transformcameraRtC2W(cameraRtC2W,RT,inverse)
         cameraRtC2Wout = cameraRtC2W;
         if ~inverse
             for i =1:size(cameraRtC2W,3)
                 cameraRtC2Wout(:,:,i) = RT(1:3,1:3)*cameraRtC2W(:,:,i);
                 cameraRtC2Wout(:,4,i) = cameraRtC2Wout(:,4,i)+RT(1:3,4);
             end
         else
             for i =1:size(cameraRtC2W,3)                              
                cameraRtC2Wout(:,4,i) = cameraRtC2W(:,4,i)-RT(1:3,4);
                cameraRtC2Wout(1:3,:,i) = RT(1:3,1:3)'*cameraRtC2Wout(1:3,:,i);
             end
         end
      
end