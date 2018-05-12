--[[
    This data loader is a modified version of the one from dcgan.torch
    (see https://github.com/soumith/dcgan.torch/blob/master/data/donkey_folder.lua).

    Copyright (c) 2016, Deepak Pathak [See LICENSE file for details]

    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.
    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
paths.dofile('dataset.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
opt.data = os.getenv('DATA_ROOT')

--------------------------------------------------------------------------------------------
local loadSize   = {opt.nc_load, opt.loadSize, opt.loadSize*opt.WtoHRatio}
local sampleSize = {opt.nc_load, opt.fineSize, opt.fineSize*opt.WtoHRatio}




local function loadImage_Height(path,nc_load)
  -- resize the image so that image height is loadSize[2] 
  local input 
  if file_exists(path) then 
    input = image.load(path , nc_load, 'float')
    local iH = input:size(2)
    local iW = input:size(3)
    -- print(('load image %d %d %d %f\n'):format(iW, iH, loadSize[2], loadSize[2]/iH*iW ))
    if loadSize[2]~=iH then
      input = image.scale(input,  loadSize[2]/iH*iW, loadSize[2])
    end
  else
      print("file not find"..path)
      input = torch.Tensor(nc_load, loadSize[2], loadSize[3])
  end
  return input
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
   collectgarbage()
   
   local input, imagename;
   if file_exists(path .. "_i_r0.4.jpg") then 
      imagename = path .. "_i_r0.4.jpg"
   else
      imagename = path .. "_i.jpg"
   end
   
    if opt.loadOpt =='rgb' then
      input = loadImage_Height(imagename,3)
      input:mul(2):add(-1)  -- make it [0, 1] -> [-1, 1]
    elseif opt.loadOpt =='sc' then
      input = loadImage_Height(path .. "_sc_r0.4.png",3) -- depth/65535
      input:mul(2):add(-1)  -- make it [0, 1] -> [-1, 1]
    elseif opt.loadOpt == 'd' then
      input = loadImage_Height(path .. "_d_r0.4.png",1) -- depth/65535
      input:mul(4):add(-1)  -- make it [0, 1] -> [-1, 1]
    elseif opt.loadOpt == 's' then
      local segMap =  loadImage_Height(path .. "_s_r0.4.png",1) 
      segMap:mul(255):round()
      input = segMap
    elseif opt.loadOpt =='pn' then
      local pImg =  loadImage_Height(path .. "_p_r0.4.png",1) -- depth/65535
      pImg:mul(4):add(-1)  
      local normalImg =  loadImage_Height(path .. "_n_r0.4.png",3) 
      normalImg:mul(2):add(-1)  
      input = torch.cat(pImg,normalImg,1)
    elseif opt.loadOpt =='pns' then
      local pImg =  loadImage_Height(path .. "_p_r0.4.png",1) -- depth/65535
      pImg:mul(4):add(-1)  
      local normalImg =  loadImage_Height(path .. "_n_r0.4.png",3) 
      normalImg:mul(2):add(-1)  
      local segMap =  loadImage_Height(path .. "_s_r0.4.png",1) 
      segMap:mul(255):round()
      input = torch.cat(torch.cat(pImg,normalImg,1),segMap,1)
    elseif opt.loadOpt =='rgbd' then
      local rgbImg = loadImage_Height(imagename,3)
      rgbImg:mul(2):add(-1)  -- make it [0, 1] -> [-1, 1]
      local depthImg =  loadImage_Height(path .. "_d_r0.4.png",1) -- depth/65535
      depthImg:mul(4):add(-1)  -- make it [0, 1] -> [-1, 1]
      input = torch.cat(rgbImg,depthImg,1) 
    elseif opt.loadOpt =='rgbdn' then   
      local rgbImg = loadImage_Height(imagename,3)
      rgbImg:mul(2):add(-1)  
      
      local depthImg =  loadImage_Height(path .. "_d_r0.4.png",1) -- depth/65535
      depthImg:mul(4):add(-1)  
      
      local normalImg =  loadImage_Height(path .. "_n_r0.4.png",3) 
      normalImg:mul(2):add(-1)  
      input = torch.cat(torch.cat(rgbImg,depthImg,1),normalImg,1)
    elseif opt.loadOpt =='rgbds' then   
      local rgbImg = loadImage_Height(imagename,3)
      rgbImg:mul(2):add(-1)  
      
      local depthImg =  loadImage_Height(path .. "_d_r0.4.png",1) -- depth/65535
      depthImg:mul(4):add(-1)  

      local segMap =  loadImage_Height(path .. "_s_r0.4.png",1) 
      segMap:mul(255):round()
      input = torch.cat(torch.cat(rgbImg,depthImg,1),segMap,1)
    elseif opt.loadOpt =='rgbdsn' then   
      
      local rgbImg = loadImage_Height(imagename,3)
      rgbImg:mul(2):add(-1)  
      
      local depthImg =  loadImage_Height(path .. "_d_r0.4.png",1) -- depth/65535
      depthImg:mul(4):add(-1)  

      local segMap =  loadImage_Height(path .. "_s_r0.4.png",1) 
      segMap:mul(255):round()

      local normalImg =  loadImage_Height(path .. "_n_r0.4.png",3) 
      normalImg:mul(2):add(-1)  
      input = torch.cat(torch.cat(torch.cat(rgbImg,depthImg,1),segMap,1),normalImg,1)
    elseif opt.loadOpt =='rgbpn' then   
      local rgbImg = loadImage_Height(imagename,3)
      rgbImg:mul(2):add(-1)  
      
      local pImg =  loadImage_Height(path .. "_p_r0.4.png",1) -- depth/65535
      pImg:mul(4):add(-1)  

      local normalImg =  loadImage_Height(path .. "_n_r0.4.png",3) 
      normalImg:mul(2):add(-1)  
      input = torch.cat(torch.cat(rgbImg,pImg,1),normalImg,1)
    elseif opt.loadOpt =='rgbpsn' then   
      local rgbImg = loadImage_Height(imagename,3)
      rgbImg:mul(2):add(-1)  
      
      local pImg =  loadImage_Height(path .. "_p_r0.4.png",1) -- depth/65535
      pImg:mul(4):add(-1)  

      local segMap =  loadImage_Height(path .. "_s_r0.4.png",1) 
      segMap:mul(255):round()

      local normalImg =  loadImage_Height(path .. "_n_r0.4.png",3) 
      normalImg:mul(2):add(-1)  
      input = torch.cat(torch.cat(torch.cat(rgbImg,pImg,1),segMap,1),normalImg,1)
    elseif opt.loadOpt =='rgbpns' or  opt.loadOpt == 'rgbpnspns' then   
      local rgbImg = loadImage_Height(imagename,3)
      rgbImg:mul(2):add(-1)  
      
      local pImg =  loadImage_Height(path .. "_p_r0.4.png",1) -- depth/65535
      pImg:mul(4):add(-1)  

      local segMap =  loadImage_Height(path .. "_s_r0.4.png",1) 
      segMap:mul(255):round()

      local normalImg =  loadImage_Height(path .. "_n_r0.4.png",3) 
      normalImg:mul(2):add(-1)  
      input = torch.cat(torch.cat(torch.cat(rgbImg,pImg,1),normalImg,1),segMap,1)
    end
   collectgarbage()
   
   -- if mutli-layer depth load other infor
   local mlinput
   if opt.loadOpt == 'rgbpnspns' then 
      local prImg =  loadImage_Height(path .. "_pr_r0.4.png",1) -- depth/65535
      prImg:mul(4):add(-1)  

      local normalrImg =  loadImage_Height(path .. "_nr_r0.4.png",3) 
      normalrImg:mul(2):add(-1)  

      local segrMap =  loadImage_Height(path .. "_sr_r0.4.png",1) 
      segrMap:mul(255):round()

      
      mlinput = torch.cat(torch.cat(prImg,normalrImg,1),segrMap,1)
     
   end

   -- crop only when necessary 
   local iW = input:size(3)
   local iH = input:size(2)
   local oW = sampleSize[3]
   local oH = sampleSize[2]
   if iH~=oH then     
      h1 = math.ceil(torch.uniform(1e-2, iH-oH))
   end
    
   if iW~=oW then
      w1 = math.ceil(torch.uniform(1e-2, iW-oW))
   end
   if iH ~= oH or iW ~= oW then 
      input = image.crop(input, w1, h1, w1 + oW, h1 + oH)
      if mlinput then 
         mlinput = image.crop(mlinput, w1, h1, w1 + oW, h1 + oH)
      end
   end
   if opt.loadOpt == 'rgbpnspns' then 
    out = torch.cat(input,mlinput,1)
   else
    out = input
   end
   assert(out:size(3) == oW)
   assert(out:size(2) == oH)
   return out
end

--------------------------------------
-- trainLoader
print('Creating train metadata')
trainLoader = dataLoader{
  paths = {opt.data},
  loadSize = {opt.nc_load, loadSize[2], loadSize[3]},
  sampleSize = {opt.nc_load, sampleSize[2], sampleSize[3]},
  split = 100,
  loadType = opt.loadType,
  serial_batches = opt.serial_batches, 
  pickbyclass = opt.pickbyclass,
  dataPath =  opt.dataPath,
  verbose = true
}
trainLoader.sampleHookTrain = trainHook
-- end
collectgarbage()

-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")
end
