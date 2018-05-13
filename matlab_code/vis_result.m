function vis_result(foldername,subfoldername,path2checkpoint,ratio,autoexit, seg_only, save_visual)
dbstop if error
addpath('./utils/');
addpath('./emd/');
addpath('./pano_io/');

if ~exist('save_visual','var')
   save_visual = 0;
end
if ~exist('foldername','var')
   foldername = 'suncg_baselinefix_pns_twoview_pns';
   subfoldername = 'mp';
   autoexit = 0;
   save_visual =1;                        
   seg_only = 0;
end

view_width = 160;
view_height = 256;
f_vir = 80;
K_vir = [f_vir 0  view_width/2;
         0 f_vir view_height/2;
         0 0 1];

%path2checkpoint = '../torch_code/checkpoints/';
folderpath = fullfile(path2checkpoint, foldername,subfoldername);
folder_inputpath = fullfile(path2checkpoint,subfoldername);
fprintf('%s\n',folderpath)
allfile = dir([folderpath '/hdf5output/*pred.h5']);

if save_visual
   savepoint = 1;
   drawbox = 1;
   drawpoint = 1;
else
   savepoint = 0;
   drawbox = 0;
   drawpoint = 0;
end

predict_sc = 0;
metaData;
objclass = pano13class;
nClass = length(objclass);

in_pId = 4;
in_segId = 8;
in_normalId = 5:7;
no_color_output = 1;

if no_color_output
    out_pId = in_pId-3;
    out_normalId = in_normalId-3;
    out_segId = [in_segId:in_segId+nClass]-3;
else
    out_pId = in_pId;
    out_segId = in_segId:in_segId+nClass;
    out_normalId = in_normalId;
end

if seg_only
   out_segId = 1:1+nClass;
end

predmask = ones(256,640);
predmask(:,size(predmask,2)/4:size(predmask,2)/4*3) = 0;
predmask = ~predmask;
gap = size(predmask,2)/4;


outputfolder_name = 'output_many';
mkdir(fullfile(folderpath,outputfolder_name))
mkdir(fullfile(folderpath,'output_ply'))
points3d_map = zeros(256,640,3);

cnt = 1;
has_d = 1;
for file_id = 1:length(allfile)
    fprintf('%d\n',file_id);
    filename = allfile(file_id).name;
    data_pred = double(hdf5read(fullfile(folderpath,'hdf5output/',filename),'result'));
    try 
        room_preds = hdf5read(fullfile(folderpath,'hdf5output/',filename),'room_est'); 
        [room_conf,room_est_batch] = max(room_preds,[],1);
        room_est(cnt) = room_est_batch;
        room_pred = 1;
    catch
        room_pred = 0;
    end   
    
    for batchId = 1:size(data_pred,4)
        try 
            color = permute((data_pred(:,:,1:3,batchId)+1)*0.5,[2,1,3]);
            has_i = 1;
        catch
            color = ones(256,640,3);
            has_i = 0;
        end
        
        if has_d
            try
                depth = permute((data_pred(:,:,out_pId,batchId)+1)*0.25*65535*0.25*0.001,[2,1]);
                has_d = 1;
            catch
                has_d = 0;
            end
        end
        
        try 
            normal = permute((data_pred(:,:,out_normalId,batchId)+1)*0.5,[2,1,3]);
            [~,points3d] = panoimgPlane2point(color,depth,normal*2-1, 0.4, [0,0,0]',3);
            points3d_mat = cell2mat(points3d);
            
            normal = reshape(normal,[],3)';
            point_vector = points3d_mat./repmat(sum(points3d_mat.^2).^0.5,[3,1]);
            normal = 2*normal-1;
            sign = sum(normal.*point_vector)>0;
            normal(repmat(sign,[3,1])) = -1*normal(repmat(sign,[3,1]));
            
            normal = reshape(normal',[size(depth,1),size(depth,2),3]);
            normal = 0.5*(normal+1);
            has_n = 1;
        catch
            % if the network don't compute surface normal, we can still
            % compute it from the predicted depth map
            if has_d
               [~,points3d] = panoimg2point(color,depth, [0,0,0]',K_vir);
               points3d_mat = cell2mat(points3d);
               normal = points2normals(points3d_mat,10);
               
               point_vector = points3d_mat./repmat(sum(points3d_mat.^2).^0.5,[3,1]);
               sign = sum(normal.*point_vector)>0;
               normal(repmat(sign,[3,1])) = -1*normal(repmat(sign,[3,1]));
               
               normal = reshape(normal',[size(depth,1),size(depth,2),3]);
               normal = 0.5*(normal+1);
               has_n = 1;
            else
               has_n = 0;
            end
        end
        
        if has_d
           for i = 1:4 
               points3d_map(:,1+gap*(i-1):gap*i,:) =reshape(points3d{i}',[size(predmask,1),size(predmask,2)/4,3]);
           end
        end
        
        try 
            seg_out = round(permute(data_pred(:,:,out_segId,batchId),[2,1,3]));
            [conf,seg] = max(seg_out,[],3);
            seg = seg-1;
            seg_out = seg_out(:,:,2:end);
            has_s = 1;
        catch
            has_s = 0;
        end
        
        
       
       
       %% write into images
       if has_i 
          imwrite(color,fullfile(folderpath, outputfolder_name, ['/color_' filename(1:end-7) num2str(batchId) '_ot.jpg']));
       end
       if has_n 
          imwrite(normal,fullfile(folderpath, outputfolder_name, ['/normal_' filename(1:end-7) num2str(batchId) '_ot.jpg']));
       end
       
       if has_d
          imwrite(getImagesc(depth),fullfile(folderpath, outputfolder_name, ['/depth_' filename(1:end-7) num2str(batchId) '_ot.jpg']));
       end
        
       if has_s 
          segIm =IdMap2colorMap(seg);
          imwrite(segIm,fullfile(folderpath, outputfolder_name, ['/seg_' filename(1:end-7) num2str(batchId) '_ot.jpg']));
       end
      
       allnamaes{cnt} = [filename(1:end-7)  num2str(batchId)];
        
        %%
        if (savepoint||drawpoint) && has_d
           [depth_r,normal_r,~,all_assigned] = pn2Room(depth,seg,normal);
           [~,points3d_r] = panoimgPlane2point(IdMap2colorMap(seg),depth_r,normal_r*2-1, 0.4, [0,0,0]',2);
           p = cell2mat(points3d_r);
           s = reshape(IdMap2colorMap(seg), [], 3);       
            if drawpoint
            f = figure(1); clf;
            if has_s
                pcshow(pointCloud(p(:,all_assigned>0&seg(:)>1)','Color', s(seg(:)>1&all_assigned>0,:)))
                hold on;
                if drawbox
                   BBpred = segPoints2boxes(reshape(p',[size(seg,1) ,size(seg,2),3]),seg,all_assigned,[4:nClass]);
                   for i =1: length(BBpred)
                       vis_cube(BBpred(i), IdMap2colorMap(BBpred(i).classId), 5);
                   end
                end
            else
                vis_point_cloud(cell2mat(points3d)',[],[],10000); 
            end
            
            
            imfilename = fullfile(folderpath, outputfolder_name, ['/pt_' filename(1:end-7) num2str(batchId) '_ot.jpg']);
            ax = gca;
            adjustImageSave(f, ax, imfilename)
           end
        end
        cnt = cnt+1;
    end
end
%% gen webpage 
fid = fopen(fullfile(folderpath,outputfolder_name,'index.html'),'w');
h = 200;

for i = 1:length(allnamaes)
    fprintf(fid, '<img src="%s"  style="height:%dpx;">\n', ['color_' allnamaes{i}   '_ot.jpg'],h);
    fprintf(fid, '<img src="%s"  style="height:%dpx;">\n', ['depth_' allnamaes{i}  '_ot.jpg'],h);
    fprintf(fid, '<img src="%s"  style="height:%dpx;">\n', ['normal_' allnamaes{i}  '_ot.jpg'],h);
    fprintf(fid, '<img src="%s"  style="height:%dpx;">\n', ['seg_' allnamaes{i}  '_ot.jpg'],h);
    fprintf(fid, '<img src="%s"  style="height:%dpx;">\n', ['pt_' allnamaes{i}  '_ot.jpg'],h);
    fprintf(fid, '<br>\n');
    
    fprintf(fid, '%d %s <br>\n',i, allnamaes{i});
    fprintf(fid, '<br>\n');
    fprintf(fid, 'depth diff: %f\n', eval(i).sm_dis_error(1)); 
    fprintf(fid, '<br>\n');
    fprintf(fid, '<br>\n');
end
fclose(fid);

if autoexit
    exit;
end