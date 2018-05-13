require 'torch'
require 'image'
require 'im2pano3d_funcs'
require 'cudnn'
require 'nngraph'
require 'cunn'
require 'nn'
require 'hdf5'

util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
   dataPath = '',
   batchSize = 16,         -- number of samples to produce
   net = '',               -- path to the generator network
   netname = '',           -- path to the generator network
   loadSize = 256,         -- resize the loaded image to loadsize maintaining aspect ratio. 0 means don't resize. -1 means scale randomly between [0.5,2] -- see donkey_folder.lua
   fineSize = 256,         -- size of random crops. Only 64 and 128 supported.
   WtoHRatio = 2.5,        -- size of random crops. Only 64 and 128 supported.
   nBottleneck = 4000,     -- #  of dim for bottleneck of encoder
   nef = 64,               -- #  of encoder filters in first conv layer
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nc = 3,                 -- # of channels in input
   wtl2 = 0.99,            -- 0 means don't use else use with this weight
   useOverlapPred = 1,     -- overlapping edges (1 means yes, 0 means no). 1 means put 10x more L2 weight on unmasked region.
   nThreads = 1,           -- #  of data loading threads to use
   checkpoints_dir = './checkpoints', -- models are saved here
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 2,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   maskType = 'center',    -- mask Type 
   name = 'mp',            -- name for save images
   loadType = 'list',      -- folder or list, always list
   loadOpt  = 'rgbpns',    -- load data type
   predroomType = true,    -- predicting room type?
   dataset = 'mp',         -- 'mp' for matterport3D data, 'suncg' for suncg data
   phase = 'test',         -- train or test
   loss_xyz = true,        -- use PNlayer
   how_many = 480,         -- how many testing image to test 
   maskangle = 180,        -- input image mask fov
   ---- don't change following parameters
   serial_batches = 1,     -- always 1
   useroomType = false,    -- always false
   pickbyclass = 0,        -- always 0
   nClasses = 13,          -- num of object class after mapping
   nClasses_load = 40,     -- num of object class while loading
   singleObjectTrain = false, -- always false
   numofboxes = 1,
   nz = 10

}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
opt = processOpt(opt)
opt.name = opt.name 
if opt.gpu>0 then
   cutorch.setDevice(opt.gpu)
end

opt.net = paths.concat(opt.checkpoints_dir, opt.name, opt.netname ..'.t7')



-- set seed
opt.manualSeed = torch.random(1, 10000)
print("Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
print(opt)

-- load 
assert(opt.net ~= '', 'provide a generator model')
print(opt.net)
net = util.load(opt.net, opt)
net:evaluate()
print(net)


local loadSize   = {opt.nc, opt.loadSize, opt.loadSize*opt.WtoHRatio}
local sampleSize = {opt.nc, opt.fineSize, opt.fineSize*opt.WtoHRatio}
local input_ctx  = torch.Tensor(opt.batchSize, opt.nc, sampleSize[2], sampleSize[3])
local room_vec   = torch.Tensor(opt.batchSize, opt.nz, 1, 1)

local TwFile = hdf5.open('Tw.h5', 'r');
local Twdata = TwFile:read('Tw'):all();
local Twdata_gobal = torch.Tensor(opt.batchSize, sampleSize[2], sampleSize[3],3)
torch.repeatTensor(Twdata_gobal,Twdata,opt.batchSize,1,1,1)
local image_ctx = torch.Tensor(opt.batchSize, opt.nc, sampleSize[2], sampleSize[3])

assert(opt.gpu > 0, 'only GPU mode supported')
input_ctx = input_ctx:cuda()
room_vec  = room_vec:cuda()
Twdata_gobal = Twdata_gobal:cuda()


-- load data
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())
paths.mkdir(paths.concat(opt.checkpoints_dir, opt.name, opt.dataset,'hdf5output'))
local cmd = 'rm -f '.. paths.concat(opt.checkpoints_dir, opt.name, opt.dataset,'hdf5output') ..'/*.h5'
print(cmd); os.execute(cmd)
paths.mkdir(paths.concat(opt.checkpoints_dir, opt.dataset, 'hdf5output'))

for n =1,math.ceil(math.min(opt.how_many,data:size())/opt.batchSize) do
    print('processing batch ' .. n .. '/')
    
    local image_ctx_load, image_class, filepaths_curr = data:getBatch() -- original input image

    image_ctx:copy(image_ctx_load[{{},{1,opt.nc},{},{}}])

    filepaths_curr = util.basename_batch(filepaths_curr)

    if opt.has_s and opt.nClasses<40 and opt.loadOpt ~= 'rgbpn' then 
       image_ctx = mapSegmap(image_ctx,opt,opt.idx_s)
    end

    if opt.has_s then 
       image_ctx[{{},opt.idx_s,{},{}}]:add(1)
    end
    
    real_center = image_ctx:clone() -- copy by value
    local mask_single = genMask(sampleSize,opt.maskType, opt)
    local mask_global = torch.ByteTensor(opt.batchSize, sampleSize[2], sampleSize[3])
    if opt.maskType == 'fillin' then 
       fillinMask(mask_global,real_center, opt)
    elseif opt.singleObjectTrain then
       singleObjectMask(mask_global,opt,mask_single,real_center)
    else
       torch.repeatTensor(mask_global,mask_single,opt.batchSize,1,1)
    end

    image_ctx = maskInput(image_ctx, mask_global, opt) -- fill masked region with mean value
    input_ctx:copy(image_ctx)

    local output_net,output_full,room_est

    if opt.useroomType then
      for i = 1,opt.batchSize do room_vec[i]:fill(image_class[i]) end
      output_net = net:forward({input_ctx,room_vec})
    else
      if opt.loss_xyz then 
         output_net = net:forward({input_ctx,Twdata_gobal})
      else
         output_net = net:forward(input_ctx)
      end
    end

    local outputsize = output_net[1]:size()
    if net.netinfo.out_idx_i then 
      if net.netinfo.out_idx_i>0 then 
        output_full = output_net[net.netinfo.out_idx_i]:float()
      else
        output_full = output_net:float()
      end
    end

    if  net.netinfo.out_idx_p then 
        if net.netinfo.out_idx_p>0 then 
          if output_full then  
             output_full = torch.cat(output_full,output_net[net.netinfo.out_idx_p]:float(),2)
          else 
             output_full = output_net[net.netinfo.out_idx_p]:float()
          end
        else
          output_full = output_net:float()
        end
    end
    
    if net.netinfo.out_idx_n then 
       local normal_est = output_net[net.netinfo.out_idx_n]
       normal_est = normal_est:view(outputsize[1], outputsize[4], outputsize[3], 3);
       normal_est = normal_est:transpose(2,4)
       output_full = torch.cat(output_full, normal_est:float(),2)
    end

    if net.netinfo.out_idx_s then 
      -- local segmentation_conf, segmentation_est =  torch.max(output_net[net.netinfo.out_idx_s]:float(), 2)
      -- local segmentation_est_covert = torch.Tensor(segmentation_est:size()):copy(segmentation_est)
      -- local SSMLayer = nn.SpatialSoftMax()
      -- SSMLayer:cuda()
      -- local segmentation_estN = SSMLayer:forward(output_net[net.netinfo.out_idx_s])
      if output_full then  
         output_full = torch.cat(output_full, output_net[net.netinfo.out_idx_s]:float(),2)
      else
         output_full = output_net[net.netinfo.out_idx_s]:float()
      end
    end


    
    if net.netinfo.out_idx_rm then 
       room_est = output_net[net.netinfo.out_idx_rm]
    end
    print('Prediction: size: ', output_full:size(1)..' x '..output_full:size(2) ..' x '..output_full:size(3)..' x '..output_full:size(4))

    -- save as hdf5
    for i= 1, opt.batchSize do
      -- print(n..i..filepaths_curr[i])
      local pred_filename = paths.concat(opt.checkpoints_dir, opt.name, opt.dataset, 'hdf5output', filepaths_curr[i]..'_pred.h5')
      local myFile = hdf5.open(pred_filename, 'w')
      myFile:write('result', output_full[i]:float())
      myFile:write('mask_global', mask_global[i]:float())
      if net.netinfo.out_idx_rm  then myFile:write('room_est', room_est[i]:float()) end
      myFile:close()
      
      if opt.saveinput then 
        local input_filename = paths.concat(opt.checkpoints_dir, opt.dataset, 'hdf5output', filepaths_curr[i]..'_input.h5')
        local myFile = hdf5.open(input_filename,  'w')
        myFile:write('result', real_center[i])
        local room_class = torch.Tensor(1):fill(image_class[i])
        myFile:write('room', room_class)
        myFile:close()
      end
    end
end
print('Saved predictions to:', paths.concat(opt.checkpoints_dir, opt.name, opt.dataset))
net = nil

local cmd 
if opt.loadOpt == 's' then 
  cmd = 'cd ../matlab_code; matlab -nosplash -nodesktop -nodisplay -r \"vis_result(\''..opt.name..'\',\''.. opt.dataset ..'\',\''.. opt.checkpoints_dir ..'\',\''..opt.maskangle..'\',1,1)\"'
else
  cmd = 'cd ../matlab_code; matlab -nosplash -nodesktop -nodisplay -r \"vis_result(\''..opt.name..'\',\''.. opt.dataset ..'\',\''.. opt.checkpoints_dir ..'\',\''..opt.maskangle..'\',1,0)\"'
end
print(cmd)
os.execute(cmd)