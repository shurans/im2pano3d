require 'nn'
require 'cudnn'
require 'cunn'
require 'nngraph'
require 'image'
require 'hdf5'



function  processOpt(opt)
        if opt.display == 0 then opt.display = false end
        if opt.conditionAdv == 0 then opt.conditionAdv = false end
        if opt.noiseGen == 0 then opt.noiseGen = false end
        if opt.useroomType == 0 then  opt.useroomType = false end
        if opt.predroomType == 0 then  opt.predroomType = false end
        if opt.singleObjectTrain == 0 then  opt.singleObjectTrain = false end
        if opt.loss_xyz == 0 then  opt.loss_xyz = false end
        if opt.inc_learning == 0 then opt.inc_learning = false end
        if opt.use_GAN == 0 then opt.use_GAN = false end

        
        -- control what loss to use for segmentation 
        if opt.distrib_loss == 0 then opt.distrib_loss = false end
        if opt.iou_loss == 0 then opt.iou_loss = false end
        if opt.segsoftmax_loss == 0 then opt.segsoftmax_loss = false end
        if opt.exist_loss == 0 then opt.exist_loss = false end
        if opt.multi_layer_d == 0 then opt.multi_layer_d = false end
        if opt.use_Unet == 0 then opt.use_Unet = false end

        local mp_maping = {3,2,5,10,8,13,11,13,4,7,6,4,11,13,12,2,1,13,5,13,13,9,12,3,12,11,3,13,3,3,12,4,12,5,3,12,13,13,13,13}
        local suncg_maping = {3,2,11,6,5,7,8,10,4,12,13,11,4,8,12,4,11,13,13,2,13,1,13,13,9,13,13,12,13,13,13,11,13,12,13,12,13,3,12,13}
        local no_mapping = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40}
        
        if opt.dataset == 'mp' then 
           opt.obj_maping = mp_maping
           print('using mp mapping')
        elseif opt.dataset == 'suncg' or opt.dataset == 'nyu' then
           opt.obj_maping = suncg_maping
           print('using suncg and nyu mapping')
        elseif opt.dataset == 'none' then
           opt.obj_maping = no_mapping
        end

        if opt.multi_layer_d then 
           opt.loadOpt = 'rgbpnspns'
        end

        
        opt.idx_rgbd = {1,4}
        opt.idx_i = {1,3}
        opt.idx_d = {4}
        opt.idx_p = {4}

        if opt.loadOpt == 'rgb' or opt.loadOpt == 'sc' then 
            opt.idx_rgbd = {1,3}
            opt.nc = 3 
            opt.nc_load = 3
            opt.has_i = true
            opt.has_d = false
            opt.has_n = false
            opt.has_s = false
            opt.has_p = false
            opt.loss_xyz = false
        end

        if opt.loadOpt == 'd' then 
            opt.idx_rgbd = {1}
            opt.idx_d = {1}
            opt.idx_p = {1}
            opt.nc = 1 
            opt.nc_load = 1
            opt.has_i = false
            opt.has_d = true
            opt.has_n = false
            opt.has_s = false
            opt.has_p = false
            opt.loss_xyz = false
        end

        if opt.loadOpt == 'pns' then 
            opt.idx_rgbd = {1}
            opt.idx_d = {1}
            opt.idx_p = {1}
            opt.idx_n = {2,4}
            opt.idx_s = {5}
            opt.nc = 5 
            opt.nc_load = 5
            opt.has_i = false
            opt.has_d = false
            opt.has_n = true
            opt.has_s = true
            opt.has_p = true
            opt.loss_xyz = true
        end

        if opt.loadOpt == 'pn' then 
            opt.idx_rgbd = {1}
            opt.idx_d = {1}
            opt.idx_p = {1}
            opt.idx_n = {2,4}
            opt.nc = 4
            opt.nc_load = 4
            opt.has_i = false
            opt.has_d = false
            opt.has_n = true
            opt.has_s = false
            opt.has_p = true
            opt.loss_xyz = true
        end
       

         if opt.loadOpt == 'rgbpn'  then 
            opt.nc = 7 
            opt.nc_load = 7
            opt.has_i = true
            opt.has_d = false
            opt.has_n = true
            opt.has_s = true -- not loading s but still predict s
            opt.has_p = true
            opt.loss_xyz = true
            opt.idx_n = {5,7}
        end
        

        if opt.loadOpt == 'rgbpns' then 
            opt.nc = 8
            opt.nc_load = 8
            opt.has_i = true
            opt.has_d = false
            opt.has_n = true
            opt.has_s = true
            opt.has_p = true
            opt.idx_n = {5,7}
            opt.idx_s = {8}
            if opt.Gtype_out == 'rgb' then 
               opt.loss_xyz = false 
            else
               opt.loss_xyz = true
            end
        end

       

        if opt.loadOpt == 's' then 
            opt.nc = 1
            opt.nc_load = 1
            opt.has_i = false
            opt.has_d = false
            opt.has_n = false
            opt.has_s = true
            opt.has_p = false
            opt.idx_s = {1}
            opt.loss_xyz = false
        end
        

        if opt.inc_learning then
           opt.singleObjectTrain = true
        end
        
        

        if opt.phase == 'train' then 
          opt.serial_batches = 0 
        elseif opt.phase == 'test' then
          opt.serial_batches = 1
        end

        if opt.netname == '' then 
           opt.netname = 'latest_net_G'
        end

        return opt
end

function mapSegmap(real_full,opt,idx_s)
      local segmask = torch.Tensor(real_full:size(1),1, real_full:size(3),real_full:size(4)):fill(0)   
      segmask:copy(real_full[{{},idx_s,{},{}}])

      for cid = 1,opt.nClasses_load do 
        local mapid = opt.obj_maping[cid]
        real_full[{{},idx_s,{},{}}][torch.eq(segmask,cid)] = mapid
      end
      return real_full
end

function  getSegMap2Vol(nClasses,segmask,input_segvol)
    -- segmask:   #batchx1xHxW
    -- input_segvol: #batchxnClassxHxW

    local input_segvol = torch.Tensor(segmask:size(1),nClasses,segmask:size(3),segmask:size(4)):fill(0)   
    for cid = 1,nClasses do 
        input_segvol[{{},{cid},{},{}}][torch.eq(segmask,cid)] = 1
    end
    return input_segvol
end

function  getSegMap2VolScaleNorm(nClasses, segmentation_est_norm, segconf, segmask, input_segvol)
    input_segvol:copy(segmentation_est_norm)
    local maxconf = 0.6
    

    local maxconfMAtrix = segconf:clone()
    maxconfMAtrix[torch.lt(maxconfMAtrix,maxconf)] = maxconf;

    local orginalconf = torch.Tensor(maxconfMAtrix:size()):fill(0):cuda()   
    for cid = 1,nClasses do
        orginalconf[torch.eq(segmask,cid)] = segmentation_est_norm[{{},{cid},{},{}}][torch.eq(segmask,cid)] 
    end
    orginalconf[torch.gt(orginalconf,0.9999)] = 0.9999

    local scaleMatrix = (1-maxconfMAtrix):cdiv(1-orginalconf)

    for cid = 1,nClasses do 
        input_segvol[{{},{cid},{},{}}]:cmul(scaleMatrix:cuda())
        input_segvol[{{},{cid},{},{}}][torch.eq(segmask,cid)] = maxconfMAtrix[torch.eq(segmask,cid)]
    end
    return input_segvol
end



function getObjDistribution(nClasses,segmask)
    local distribution = torch.Tensor(segmask:size(1),nClasses):fill(0) 
    for bid = 1,segmask:size(1) do
        for cid = 1,nClasses do
            distribution[bid][cid] = torch.sum(torch.eq(segmask[bid],cid))
        end
    end
    local distributionS = torch.sum(distribution,2)
    for bid = 1,segmask:size(1) do
        distribution[bid] = distribution[bid]:div(distributionS[bid][1])
    end
    return distribution
end
function getObjExistence(nClasses,segmask,mask_single)
    local existence = torch.Tensor(segmask:size(1),nClasses):fill(0) 
    for bid = 1,segmask:size(1) do
        for cid = 1,nClasses do
            local thisclass_inmask = torch.eq(segmask[bid],cid):cmul(mask_single)
            existence[bid][cid] = torch.sum(thisclass_inmask)
        end
    end
    existence = torch.gt(existence,400):float()

    return existence
end

function  getBBfromMask(objectmask,minsize)
    local miny,maxy,minx,maxx = 0,0,0,0
    local ycnt = 0
    local ycrood = {}
    local bbcnt = 0
    local bb = {}
    
    for yi = 1,objectmask:size(1) do 
        if torch.any(objectmask[{{yi},{}}]) then 
            if miny ==0 then 
               miny = yi 
            else
               maxy = yi
            end
        elseif miny>0 and maxy>0 then
            if (maxy-miny) > minsize then
                ycnt = ycnt+1 
                ycrood[ycnt] = {miny,maxy}
            end
            miny = 0
            maxy = 0
        end
    end
    
    for j = 1,ycnt do
        for xi = 1,objectmask:size(2) do 
            if torch.any(objectmask[{ycrood[j],{xi}}]) then 
                if minx ==0 then 
                   minx = xi 
                else
                   maxx = xi
                end
            elseif minx>0 and maxx>0 then
                if (maxx-minx) > minsize then
                   bbcnt = bbcnt+1
                   bb[bbcnt] = {ycrood[j][1],ycrood[j][2],minx,maxx}
                end
                minx = 0
                maxx = 0
            end
        end
    end

    if bbcnt==0 then
       bb = nil
    else
       bb = torch.Tensor(bb) 
    end

    return bb
end

function getBBfromMaskRand(mask_single)
    local h = mask_single:size(1);
    local w = mask_single:size(2);
    local miny = torch.floor(torch.rand(1)*(h-2))+1
    local maxy = miny+torch.floor(torch.rand(1)*(h-miny-1))
    local minx = torch.floor(torch.rand(1)*(w-2))+1
    local maxx = minx+torch.floor(torch.rand(1)*(w-minx-1))
    local bb = torch.Tensor({{miny[1],maxy[1],minx[1],maxx[1]}}):round()
    

    return bb 
end

function maskInput(real_full, mask_global, opt)
    local real_ctx = torch.Tensor(real_full:size())
    real_ctx:copy(real_full)
    if opt.maskType == 'mask_1camera_pns' then 
        for i =4,real_ctx:size(2) do
            real_ctx[{{},{i},{},{}}][mask_global] = 0
        end
    else
        if opt.has_i then 
            real_ctx[{{},{1},{},{}}][mask_global] = 2*117.0/255.0 - 1.0
            real_ctx[{{},{2},{},{}}][mask_global] = 2*104.0/255.0 - 1.0
            real_ctx[{{},{3},{},{}}][mask_global] = 2*123.0/255.0 - 1.0
            
            for i =4,real_ctx:size(2) do
                real_ctx[{{},{i},{},{}}][mask_global] = 0
            end

            
        else
        	for i =1,real_ctx:size(2) do
                real_ctx[{{},{i},{},{}}][mask_global] = 0
            end
        end
    end

	return real_ctx
end

function fillinMask(mask_global,  real_full, opt)
    mask_global:fill(0)
    for imageId = 1,real_full:size(1) do
        local seg = real_full[{{imageId},opt.idx_s,{},{}}]:squeeze()
        mask_global[{{imageId},{}}][torch.eq(seg,1)] = 1
    end
end



function singleObjectMask(mask_global, opt, mask_single, real_full)
    mask_global:fill(0)
    local numofboxes = opt.numofboxes
    for imageId = 1,real_full:size(1) do
        local seg = real_full[{{imageId},opt.idx_s,{},{}}]:squeeze()
        local allbb = nil
        local bbcount = 0
        for classId = 1,opt.nClasses+1 do 
            if classId > 2 and  classId ~= 22 then
               local objectmask = torch.eq(seg,classId)
               objectmask[torch.eq(mask_single,0)] = 0

               if torch.any(objectmask) then 
                  local bbarray = getBBfromMask(objectmask,5)

                  if bbarray then
                     if allbb ==nil then
                       allbb = bbarray
                     else
                       allbb = torch.cat(allbb,bbarray,1)
                     end
                     
                  end
               end
            end
        end

        if allbb then
          bbcount = allbb:size(1)
        end
        if bbcount < numofboxes then 
          for i = 1,(numofboxes-bbcount) do 
              local bbrandom = getBBfromMaskRand(mask_single);
              if allbb then
                allbb = torch.cat(allbb,bbrandom,1)
              else
                allbb = bbrandom
              end
          end
        end


        local bbpick = torch.randperm(allbb:size(1))
        bbpick = bbpick:narrow(1,1,numofboxes)
                    
        for i = 1,bbpick:size(1) do
            mask_global[{{imageId},{allbb[bbpick[i]][1],allbb[bbpick[i]][2]},{allbb[bbpick[i]][3],allbb[bbpick[i]][4]}}] = 1
        end
    end
    return mask_global
end


local res = 0.01 -- the lower it is, the more continuous the output will be. 0.01 is too small and 0.1 is too large
local density = 0.5
local MAX_SIZE = 10000
local low_pattern = torch.Tensor(res*MAX_SIZE, res*MAX_SIZE):uniform(0,1):mul(255)
local pattern = image.scale(low_pattern, MAX_SIZE, MAX_SIZE,'bicubic')
low_pattern = nil
pattern:div(255);
pattern = torch.lt(pattern,density):byte()  -- density 1s and 1-density 0s
pattern = pattern:byte()
function genMask(sampleSize,masktype,opt)
	-- get mask based on option
    -- 0 i sinput 1 is output regoin 
    local mask_single = torch.ByteTensor(sampleSize[2], sampleSize[3])
	if masktype == 'twoview' then
		mask_single = mask_single:fill(0)
		mask_single[{{},{1 + sampleSize[3]/4, sampleSize[3]/2 + sampleSize[3]/4}}]  = 1;
    elseif masktype == 'random' then
        while true do
            local x = torch.uniform(1, MAX_SIZE-sampleSize[3])
            local y = torch.uniform(1, MAX_SIZE-sampleSize[2])
            mask_single = pattern[{{y,y+sampleSize[2]-1},{x,x+sampleSize[3]-1}}]  -- view, no allocation
            local area = mask_single:sum()*100./(sampleSize[2]*sampleSize[3])
            local wastedIter = 0
            if area>20 and area<80 then
                break
            end
            wastedIter = wastedIter + 1
        end
    elseif masktype == 'nomask' then
        mask_single = mask_single:fill(0)   
    elseif masktype == 'ratio' then
        mask_single = mask_single:fill(0)
        margin = torch.round(opt.maskangle/360*sampleSize[3]/2)
        mask_single[{{},{1 + margin, sampleSize[3]- margin}}]  = 1;
    
    elseif masktype == 'middlecamera' then
        local diff_mask = hdf5.open('diff_mask.h5', 'r');
        mask_single = diff_mask:read('mask_middlecamera'):all();
    elseif masktype == 'upcamera' then
        local diff_mask = hdf5.open('diff_mask.h5', 'r');
        mask_single = diff_mask:read('mask_upcamera'):all();   
    elseif masktype == 'bottmcamera' then
        local diff_mask = hdf5.open('diff_mask.h5', 'r');
        mask_single = diff_mask:read('mask_bottmcamera'):all();   
    elseif masktype == '3camera' then
        local diff_mask = hdf5.open('diff_mask.h5', 'r');
        mask_single = diff_mask:read('mask_3camera'):all();   
    elseif  masktype == 'mask_1camera_pns' then  
        local diff_mask = hdf5.open('diff_mask.h5', 'r');
        mask_single = diff_mask:read('mask_1camera'):all();   
    elseif  masktype == '1camera' then  
        local diff_mask = hdf5.open('diff_mask.h5', 'r');
        mask_single = diff_mask:read('mask_1camera'):all();   
    else
        print('wrong masktype ..')
    end
	return mask_single;
end




