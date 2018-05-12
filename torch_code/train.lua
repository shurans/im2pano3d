require 'torch'
require 'optim'
require 'image'
require 'cudnn'
require 'hdf5'
require 'cunn'
require 'im2pano3d_funcs'
require 'im2pano3d_models'


util = paths.dofile('util.lua')

opt = {
   batchSize = 3,         -- number of samples to produce
   loadSize = 256,         -- resize the loaded image to loadsize maintaining aspect ratio. 0 means don't resize. -1 means scale randomly between [0.5,2] -- see donkey_folder.lua
   fineSize = 256,         -- size of random crops. Only 64 and 128 supported.
   WtoHRatio = 2.5,         -- size of random crops. Only 64 and 128 supported.
   nBottleneck = 4000,      -- #  of dim for bottleneck of encoder
   nef = 64,               -- #  of encoder filters in first conv layer
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 32,               -- #  of discrim filters in first conv layer
   nc = 3,                 -- # of channels in input           
   useOverlapPred = 1,     -- overlapping edges (1 means yes, 0 means no).  1 means put overlapL2Weightx L2 weight on unmasked region.
   overlapL2Weight = 1,    -- overlapL2Weightx L2 weight on unmasked region. 
   nThreads = 1,           -- #  of data loading threads to use
   save_epoch_freq = 5,    -- save a model every save_epoch_freq epochs (does not overwrite previously saved models)
   save_latest_freq = 1000,-- save the latest model every latest_freq sgd iterations (overwrites the previous latest model)
   checkpoints_dir = './checkpoints', -- models are saved here
   niter = 100,            -- #  of iter at starting learning rate
   lr = 0.0001,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   display_iter = 10,      -- # number of iterations after which display is updated
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'train1',        -- name of the experiment you are running
   dataset = 'mp',
   nClasses_load = 40,     -- num of object class when loading the data 
   nClasses = 13,          -- num of object class after conversion
   phase    = 'train',     -- train or test
   serial_batches = 0,     -- if 1, takes images in order to make batches, otherwise takes them randomly
   ftGModel = '',          -- path to generator model
   ftDModel = '',          -- path to discriminator  model
   ftModel = '',           -- path to the folder contains G and D
   use_Unet = true,        -- always true
   use_GAN = true,         -- use GAN?
   continue_train = 1,      -- load existing model and continue training 
   maskType  = 'twoview',   -- option for mask type 
   loadType  = 'list',      -- list 
   loadOpt   = 'rgbpns',    -- option for loading data type
   Gtype_in  = 'rgbpns',    -- option input modality 
   Gtype_out = 'pns',       -- option output modality 
   nrmType = 8,            -- num of room class
   wt_GANg = 0.01,
   wt_i   = 0.99, 
   wt_d   = 0.4,           
   wt_n   = 0.4,           
   wt_s   = 0.99, 
   wt_xyz = 0.001,
   wt_d_r = 0.4, 
   wt_n_r = 0.4,   
   wt_s_r = 0.4,    
   wt_xyz_r = 0.001, 
   -- do not chage the following parameters
   Dtype  = 'eq',
   nc_out = 4,
   useroomType = false,
   pickbyclass = 0,
   dataPath = '/',
   predroomType = true,
   attribute_exist = false,
   loss_xyz = true,
   -- param to control training
   singleObjectTrain = false,
   numofboxes = 1,
   inc_learning = false,
   netD_nlayer = 3,

   -- additional loss
   wt_distrib = 0.01,
   distrib_loss  = false,
   exist_loss = false,
   segsoftmax_loss = true,
   maskangle = 180,
   nz = 10,               -- #  of dim for Z
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
opt = processOpt(opt)
opt.name = opt.name..'_'..opt.maskType ..'_'..opt.loadOpt
print(opt)
assert(opt.gpu > 0, 'only GPU mode supported')
cutorch.setDevice(opt.gpu)

opt.manualSeed = torch.random(1, 10000)
print("Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()


-- create data loader
local DataLoader =   paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())

---------------------------------------------------------------------------
-- Initialize network variables
---------------------------------------------------------------------------
local nc = opt.nc
local nz = opt.nz
local real_label = 1
local fake_label = 0

---------------------------------------------------------------------------
-- Loss Metrics
---------------------------------------------------------------------------
local BCECriterion = nn.BCECriterion()
local netDCriterion -- defined later 
local criterionMSE = nn.MSECriterion() -- L2 loss
local criterionAbs = nn.AbsCriterion() -- L1 loss
local criterionAbs_exist = nn.AbsCriterion() -- L1 loss
local criterionCOS = nn.CosineEmbeddingCriterion()
local classWeights = torch.ones(opt.nClasses+1)
classWeights[1] = 0 -- ignore the first class 
local criterionCEC = cudnn.SpatialCrossEntropyCriterion(classWeights)
local criterionLCEC = nn.CrossEntropyCriterion()
local criterionSL1 = nn.SmoothL1Criterion()
local SSMLayer = nn.SpatialSoftMax()

---------------------------------------------------------------------------
-- Define network
---------------------------------------------------------------------------
-- load saved models and finetune
if opt.ftModel ~='' then 
   opt.ftGModel = paths.concat(opt.checkpoints_dir, opt.ftModel, 'latest_net_G.t7')
   opt.ftDModel = paths.concat(opt.checkpoints_dir, opt.ftModel, 'latest_net_D.t7')
end

if opt.ftGModel ~='' then 
    print('finetuning G:'..opt.ftGModel)
    netG = util.load(opt.ftGModel,opt)
else
    netG = defineG(opt)
    netG:apply(weights_init,opt)
end

if opt.ftDModel ~='' then 
    print('finetuning D:'..opt.ftDModel)
    netD = util.load(opt.ftDModel,opt)
else
    netD = defineD(opt,netG.netinfo)
    netD:apply(weights_init)
end


if opt.continue_train == 1 and util.file_exists(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7')) then
   print('loading previously trained model ...')
   netG = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), opt)
   netD = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), opt)
end

if netG == nil and netD == nil then 
  print('define model ...')
  netG = defineG(opt)
  netG:apply(weights_init)
  netD = defineD(opt,netG.netinfo)
  netD:apply(weights_init)
end

opt.nc_out = netG.netinfo.out_dim
netDCriterion = BCECriterion

local netDis
if opt.distrib_loss then 
   netDis = define_distributionNet(opt)
   netDis = netDis:cuda()
end

local netExist
if opt.exist_loss then 
  netExist = define_existenceNet(opt)
  netExist = netExist:cuda()
end


---------------------------------------------------------------------------
-- Setup Solver
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}

---------------------------------------------------------------------------
-- Initialize data variables
---------------------------------------------------------------------------
local loadSize   = {opt.nc_load, opt.loadSize, opt.loadSize*opt.WtoHRatio}
local sampleSize = {opt.nc_load, opt.fineSize, opt.fineSize*opt.WtoHRatio}

local errD, errG, errG_l1, errG_n, errG_s, errG_r, errG_xyz, errG_distrib, errG_exist = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
local errG_xyzr, errG_nr, errG_sr, errG_pr = 0,0,0,0

local TwFile = hdf5.open('Tw.h5', 'r');
local Twdata = TwFile:read('Tw'):all();
local Twdata_gobal = torch.Tensor(opt.batchSize, sampleSize[2], sampleSize[3],3)
torch.repeatTensor(Twdata_gobal,Twdata,opt.batchSize,1,1,1)


local input_ctx = torch.Tensor(opt.batchSize, nc, sampleSize[2], sampleSize[3])
local input_full = torch.Tensor(opt.batchSize, nc, sampleSize[2], sampleSize[3])
local outputsize = input_full:size()
local output_full,wtl2Matrix
if netG.netinfo.out_idx_i and netG.netinfo.out_idx_p then 
   output_full    = torch.Tensor(opt.batchSize,4 , sampleSize[2], sampleSize[3])
   wtl2Matrix = torch.Tensor(opt.batchSize, 4, sampleSize[2], sampleSize[3])
elseif  netG.netinfo.out_idx_i then
  output_full    = torch.Tensor(opt.batchSize, 3, sampleSize[2], sampleSize[3])
  wtl2Matrix = torch.Tensor(opt.batchSize, 3, sampleSize[2], sampleSize[3])
elseif  netG.netinfo.out_idx_p then 
  output_full    = torch.Tensor(opt.batchSize,1, sampleSize[2], sampleSize[3])
  wtl2Matrix = torch.Tensor(opt.batchSize, 1, sampleSize[2], sampleSize[3])
end 


local input_ctxfull  = torch.Tensor(opt.batchSize, opt.nc_out, sampleSize[2], sampleSize[3])
local output_ctxfull = torch.Tensor(opt.batchSize, opt.nc_out, sampleSize[2], sampleSize[3])
local input_segvol   = torch.Tensor(opt.batchSize, opt.nClasses+1, sampleSize[2], sampleSize[3])



local room_est = torch.Tensor(opt.batchSize, opt.nrmType)
local room_gt  = torch.Tensor(opt.batchSize, 1)
local room_vec = torch.Tensor(opt.batchSize, nz, 1, 1)

local label = torch.Tensor(opt.batchSize)

-- additional outputs 
local normal_est
local segmentation_est,segmentation_vis,segmentation_estN
local xyzword_est


if opt.has_n then
   normal_est = torch.Tensor(opt.batchSize*sampleSize[2]*sampleSize[3], 3)
end

if opt.has_s then
   segmentation_est = torch.Tensor(opt.batchSize, opt.nClasses+1, sampleSize[2], sampleSize[3])
   segmentation_vis = torch.Tensor(opt.batchSize, 1, sampleSize[2], sampleSize[3])
end

if opt.loss_xyz then
   xyzword_est  = torch.Tensor(opt.batchSize, 3, sampleSize[2], sampleSize[3])
end


-- Initialize mask
local mask_single = genMask(sampleSize,opt.maskType,opt)
local mask_global = torch.ByteTensor(opt.batchSize, sampleSize[2], sampleSize[3])



if opt.gpu > 0 then
   Twdata_gobal = Twdata_gobal:cuda();
   input_ctx   = input_ctx:cuda();  
   input_full  = input_full:cuda();
   if output_full then 
      output_full = output_full:cuda();
      wtl2Matrix  = wtl2Matrix:cuda();
   end
   
   
   input_ctxfull  = input_ctxfull:cuda()
   output_ctxfull = output_ctxfull:cuda()

   if opt.has_n then 
    normal_est = normal_est:cuda() 
   end
   if opt.has_s then 
    segmentation_est = segmentation_est:cuda()
    segmentation_vis = segmentation_vis:cuda()
    input_segvol = input_segvol:cuda()
   end
   if opt.loss_xyz then xyzword_est = xyzword_est:cuda(); end
   
   
   room_est = room_est:cuda()
   room_vec = room_vec:cuda()
   room_gt = room_gt:cuda()
   label = label:cuda() 
       
   netG = util.cudnn(netG);     
   netD = util.cudnn(netD);
   netD:cuda();           
   netG:cuda();  
        
   criterionMSE:cuda();
   criterionAbs:cuda()
   criterionAbs_exist:cuda()
   criterionCOS:cuda();  
   criterionCEC:cuda();  
   criterionLCEC:cuda();
   criterionSL1:cuda();

   BCECriterion:cuda();  
   netDCriterion:cuda();
   SSMLayer:cuda();
end
print('NetG:',netG)
print('NetD:',netD)


local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()



-- variables used for some lose 
local distribution_gt
local segvol_gt
local exist_gt

local real_full = torch.Tensor(opt.batchSize, opt.nc, sampleSize[2], sampleSize[3])

function createRealFake()
  -- Pepare data
  data_tm:reset(); 
  local real_full_load, real_class, filepaths_curr = data:getBatch()
  real_full:copy(real_full_load[{{},{1,opt.nc},{},{}}])

  

  -- map the segmentation class
  if opt.has_s and opt.nClasses<40 then 
     real_full = mapSegmap(real_full,opt,opt.idx_s)
  end

  -- class start with 1
  if opt.has_s then 
     real_full[{{},opt.idx_s,{},{}}]:add(1)
  end

  input_full:copy(real_full)
  mask_single = genMask(sampleSize,opt.maskType)
  if opt.maskType == 'fillin' then 
     fillinMask(mask_global,real_center, opt)
  elseif opt.singleObjectTrain then
    mask_global = singleObjectMask(mask_global,opt,mask_single,real_full)
  else
    torch.repeatTensor(mask_global,mask_single,opt.batchSize,1,1)
  end
  
  --- optional lose
  if opt.distrib_loss then 
    distribution_gt = getObjDistribution(opt.nClasses+1, real_full[{{},opt.idx_s,{},{}}])
  end
  
  if opt.exist_loss or netG.netinfo.out_idx_att then
     exist_gt = getObjExistence(opt.nClasses+1, real_full[{{},opt.idx_s,{},{}}],mask_single)
  end


  local real_ctx = maskInput(real_full, mask_global, opt) 
  input_ctx:copy(real_ctx)
  room_gt:copy(real_class)
  
  if opt.useroomType then
    for i = 1,opt.batchSize do
        room_vec[i]:fill(real_class[i])
    end
  end

  --debug graph 
  -- nngraph.setDebug(true)
  -- netG.name = 'debug_graph'
  -- pcall(function() netG:updateOutput(input_ctx) end)
  -- graph.dot(netG.fg, 'Forward Graph', 'outputBasename')

  -- create fake
  local fake
  if opt.useroomType then
    fake = netG:forward({input_ctx,room_vec})
  elseif opt.loss_xyz then 
    fake = netG:forward({input_ctx,Twdata_gobal}) 
  else
    fake = netG:forward(input_ctx)
  end

  
  -- copy out prediction 
  if netG.netinfo.out_idx_i and netG.netinfo.out_idx_p then 
     output_full:copy(torch.cat(fake[netG.netinfo.out_idx_i],fake[netG.netinfo.out_idx_p],2))
     opt.outidx_rgbd = {1,4}
  elseif  netG.netinfo.out_idx_p then 
    if netG.netinfo.out_idx_p >0 then 
      output_full:copy(fake[netG.netinfo.out_idx_p])
    else
      output_full:copy(fake)
    end
     opt.outidx_rgbd = {1}
  elseif  netG.netinfo.out_idx_i then 
    if netG.netinfo.out_idx_i >0 then 
      output_full:copy(fake[netG.netinfo.out_idx_i])
    else
      output_full:copy(fake)
    end
     
     opt.outidx_rgbd = {1,3}
  end

  if netG.netinfo.out_idx_n then 
     normal_est:copy(fake[netG.netinfo.out_idx_n]) 
  end

  if netG.netinfo.out_idx_s then 
     segmentation_est:copy(fake[netG.netinfo.out_idx_s])
  end

  if opt.loss_xyz then 
     xyzword_est:copy(fake[netG.netinfo.out_idx_xyz])
  end

  if opt.predroomType then 
     room_est:copy(fake[netG.netinfo.out_idx_rm])
  end
 
  
  -- form output_ctxfull for netD
  local startInd = 1
  if netG.netinfo.out_idx_i then 
    if netG.netinfo.out_idx_i >0 then 
      output_ctxfull:narrow(2,startInd,3):copy(fake[netG.netinfo.out_idx_i ])
    else
      output_ctxfull:narrow(2,startInd,3):copy(fake)
    end
    input_ctxfull:narrow(2,startInd,3):copy(input_full[{{},opt.idx_i,{},{}}])
    opt.outidx_i  = {startInd,startInd+2}
    startInd = startInd+3
  end 

  if netG.netinfo.out_idx_p then 
    if  netG.netinfo.out_idx_p >0 then 
     output_ctxfull:narrow(2,startInd,1):copy(fake[netG.netinfo.out_idx_p])
    else
      output_ctxfull:narrow(2,startInd,1):copy(fake)
    end
    input_ctxfull:narrow(2,startInd,1):copy(input_full[{{},opt.idx_p,{},{}}])
    opt.outidx_p  = {startInd}
    startInd = startInd+1
  end

  if netG.netinfo.out_idx_n then 
     output_ctxfull:narrow(2,startInd,3):copy(normal_est:view(outputsize[1], outputsize[4], outputsize[3], 3):transpose(2,4),2)
     input_ctxfull:narrow(2,startInd,3):copy(input_full[{{},opt.idx_n,{},{}}])
     opt.outidx_n  = {startInd,startInd+2}
     startInd = startInd+3
  end

  if netG.netinfo.out_idx_s then 
     segmentation_estN = SSMLayer:forward(fake[netG.netinfo.out_idx_s])
     output_ctxfull:narrow(2,startInd,opt.nClasses+1):copy(segmentation_estN)
     local seg_conf, seg_vis = torch.max(segmentation_estN, 2)
     segmentation_vis:copy(seg_vis)
     getSegMap2VolScaleNorm(opt.nClasses+1,segmentation_estN, seg_conf,real_full[{{},opt.idx_s,{},{}}], input_segvol)
     input_ctxfull:narrow(2,startInd,opt.nClasses+1):copy(input_segvol)
     
     opt.outidx_s  = {startInd,startInd+opt.nClasses+1-1}
     startInd = startInd+opt.nClasses+1
  end
end

---------------------------------------------------------------------------
-- Define generator and adversary closures
---------------------------------------------------------------------------
-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

   gradParametersD:zero()

   -- Real 
   local output = netD:forward(input_ctxfull)
   local otSize 
   otSize = output:size()


   local label = torch.FloatTensor(otSize):fill(real_label)
   if opt.gpu>0 then 
      label = label:cuda()
   end

   local netD_label = label

   local errD_real = netDCriterion:forward(output, netD_label)
   local df_do = netDCriterion:backward(output, netD_label)
   netD:backward(input_ctxfull, df_do)

   -- Fake 
   label:fill(fake_label)
   local output = netD:forward(output_ctxfull)
   local errD_fake = netDCriterion:forward(output, netD_label)
   local df_do = netDCriterion:backward(output, netD_label)
   netD:backward(output_ctxfull, df_do)

   errD = errD_real + errD_fake

   return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

   gradParametersG:zero()

   
   local df_dg = torch.zeros(output_ctxfull:size())
   if opt.gpu>0 then 
      df_dg = df_dg:cuda();
   end

   if opt.use_GAN then
      local output = netD.output -- netD:forward(...) was already executed in fDx, so save computation
      local label = torch.FloatTensor(output:size()):fill(real_label)
      if opt.gpu>0 then label = label:cuda(); end
      local netD_label = label

      
      errG = netDCriterion:forward(output, netD_label)
      local df_do = netDCriterion:backward(output, netD_label)
      df_dg = netD:updateGradInput(output_ctxfull, df_do)
   else
      errG = 0
   end
   df_dg:mul(opt.wt_GANg)

   -- caculate Pixelwise lose
   local df_dg_l1,df_dg_n,df_dg_s,df_dg_r,df_dg_xyz
   local df_dg_pr,df_dg_nr,df_dg_sr,df_dg_pr

   -- final grad
   local grad = {}
   
   if opt.has_i or opt.has_p or opt.has_d then 
      if opt.has_i and  netG.netinfo.out_idx_i then 
        errG_l1  = criterionAbs:forward(output_full[{{},opt.outidx_rgbd,{},{}}], input_full[{{},opt.idx_i,{},{}}])
        df_dg_l1 = criterionAbs:backward(output_full[{{},opt.outidx_rgbd,{},{}}], input_full[{{},opt.idx_i,{},{}}])
      else
        errG_l1  = criterionAbs:forward(output_full[{{},opt.outidx_rgbd,{},{}}], input_full[{{},opt.idx_p,{},{}}])
        df_dg_l1 = criterionAbs:backward(output_full[{{},opt.outidx_rgbd,{},{}}], input_full[{{},opt.idx_p,{},{}}])
      end
      

      local invalidColorMap,validDepthMap
      if opt.has_i then 
        invalidColorMap = torch.sum(input_full[{{},opt.idx_rgb,{},{}}],2);
        invalidColorMap = torch.lt(invalidColorMap,-2.9999)
      end

      if opt.has_d then 
         validDepthMap = torch.gt(input_full[{{},opt.idx_d,{},{}}],-0.99)
      else
         validDepthMap = mask_global:clone():fill(1);
      end

      wtl2Matrix:fill(opt.wt_i*opt.overlapL2Weight)
      for i= 1,wtl2Matrix:size(2) do
        if i == opt.idx_p[1] or i == opt.idx_d[1] then 
           wtl2Matrix[{{},{i},{},{}}][mask_global] = opt.wt_d
        else
           wtl2Matrix[{{},{i},{},{}}][mask_global] = opt.wt_i
           wtl2Matrix[{{},{i},{},{}}][invalidColorMap] = 0
        end
        wtl2Matrix[{{},{i},{},{}}][torch.eq(validDepthMap,0)] = 0
      end
      df_dg_l1 = df_dg:narrow(2, 1, df_dg_l1:size(2))+df_dg_l1
      df_dg_l1:cmul(wtl2Matrix)
      if (netG.netinfo.out_idx_i and netG.netinfo.out_idx_i < 0) or (netG.netinfo.out_idx_p  and netG.netinfo.out_idx_p <0) then
        grad = df_dg_l1;
      else
        if netG.netinfo.out_idx_i then 
          grad[netG.netinfo.out_idx_i] = df_dg_l1[{{},opt.outidx_i,{},{}}];
        end
        if netG.netinfo.out_idx_p then 
           grad[netG.netinfo.out_idx_p] = df_dg_l1[{{},opt.outidx_p,{},{}}];
        end
      end
   end


   if netG.netinfo.out_idx_n then
      local normal_gt = input_full[{{},opt.idx_n,{},{}}]:transpose(2,4):contiguous():view(-1,3);
      errG_n   = criterionCOS:forward({normal_est, normal_gt}, torch.Tensor(normal_est:size(1)):cuda():fill(1))
      df_dg_n  = criterionCOS:backward({normal_est, normal_gt}, torch.Tensor(normal_est:size(1)):cuda():fill(1))
      df_dg_n  = df_dg_n[1]
      
      local normal_mask = mask_global:transpose(1,3):contiguous():view(-1,1):expandAs(df_dg_n)
      df_dg_n[torch.eq(normal_mask,0)] = df_dg_n[torch.eq(normal_mask,0)]:mul(opt.overlapL2Weight)
      df_dg_n:mul(opt.wt_n)

      -- add gan lose
      if opt.use_GAN then
        local df_dg_nn = df_dg:narrow(2, opt.outidx_n[1], 3):transpose(2,4):contiguous():view(-1,3)
        df_dg_n = df_dg_n+df_dg_nn
      end

      -- set invalid lose to 0 
      local validDepthMap = torch.gt(input_full[{{},opt.idx_d,{},{}}],-0.999)
      local normal_validmask = validDepthMap:transpose(2,4):contiguous():view(-1,1):expandAs(df_dg_n)
      df_dg_n[torch.eq(normal_validmask,0)] = 0

      grad[netG.netinfo.out_idx_n] = df_dg_n

   end
   
   if netG.netinfo.out_idx_s  then
      -- mask invalid segmentation ids
      -- assert(torch.max(input_full[{{},{opt.idx_s},{},{}}])<=opt.nClasses,'input segmentation wrong')
      
      local seg_notvalidmask = torch.le(input_full[{{},opt.idx_s,{},{}}],1) 
      input_full[{{},opt.idx_s,{},{}}][seg_notvalidmask] = 1;

      errG_s  = criterionCEC:forward(segmentation_est, input_full[{{},opt.idx_s,{},{}}]:squeeze(2)) 
      df_dg_s = criterionCEC:backward(segmentation_est,input_full[{{},opt.idx_s,{},{}}]:squeeze(2))

      for i =1,df_dg_s:size(2) do
          df_dg_s[{{},{i},{},{}}][torch.eq(mask_global,0)] = df_dg_s[{{},{i},{},{}}][torch.eq(mask_global,0)]:mul(opt.overlapL2Weight) -- unmark regoin = x*opt.overlapL2Weight*wt_s
      end
      df_dg_s:mul(opt.wt_s)
      
      -- add gan loss: the max loss is not right 
      if opt.use_GAN  then
         local df_dg_ss = df_dg:narrow(2, opt.outidx_s[1], opt.nClasses+1)
         local df_dg_ssM = SSMLayer:updateGradInput(segmentation_est,df_dg_ss)
         df_dg_s = df_dg_s + df_dg_ssM
      end

      if opt.distrib_loss then 
         local distribution_est = netDis:forward(segmentation_est)
         distribution_gt = distribution_gt:cuda()
         
         errG_distrib = criterionAbs:forward(distribution_est,distribution_gt)
         local df_dg_distrib_c  = criterionAbs:backward(distribution_est,distribution_gt)
         local df_dg_distrib = netDis:updateGradInput(segmentation_est,df_dg_distrib_c)
         df_dg_s = df_dg_s + df_dg_distrib:mul(opt.wt_distrib)
      end

     
      if opt.exist_loss then
         local mask_global_float = mask_global:float():cuda()
         local exist_gt = exist_gt:cuda()
         local exist_est = netExist:forward({segmentation_est,mask_global_float})
         errG_exist = criterionAbs_exist:forward(exist_est,exist_gt)
         local df_dg_exist_c  = criterionAbs_exist:backward(exist_est,exist_gt)
         local df_dg_exist = netExist:updateGradInput(exist_est,df_dg_exist_c)
         df_dg_s = df_dg_s + df_dg_exist[1]:mul(opt.wt_distrib)
      end

      -- set invalid loss to 0 
      for i =1,df_dg_s:size(2) do
          df_dg_s[{{},{i},{},{}}][seg_notvalidmask] = 0
      end

      grad[netG.netinfo.out_idx_s] = df_dg_s
   end   
   
   if netG.netinfo.out_idx_xyz then 
      local netPM = define_PNNet()
      netPM:cuda()
      local XYZ_gt  = netPM:forward({input_full[{{},opt.idx_p,{},{}}],input_full[{{},opt.idx_n,{},{}}],Twdata_gobal})
      errG_xyz  = criterionSL1:forward(xyzword_est, XYZ_gt)
      df_dg_xyz = criterionSL1:backward(xyzword_est,XYZ_gt)
      local validDepthMap = torch.gt(input_full[{{},opt.idx_d,{},{}}],-0.99)
      for i= 1,df_dg_xyz:size(2) do
          df_dg_xyz[{{},{i},{},{}}][torch.eq(validDepthMap,0)] = 0 
          df_dg_xyz[{{},{i},{},{}}][torch.eq(mask_global,0)] = df_dg_xyz[{{},{i},{},{}}][torch.eq(mask_global,0)]:mul(opt.overlapL2Weight) 
      end
      df_dg_xyz:mul(opt.wt_xyz)
      grad[netG.netinfo.out_idx_xyz] = df_dg_xyz
   end

   if netG.netinfo.out_idx_rm then 
      errG_r  = criterionLCEC:forward(room_est, room_gt) 
      df_dg_r = criterionLCEC:backward(room_est,room_gt)
      grad[netG.netinfo.out_idx_rm] = df_dg_r
   end

  
   collectgarbage()
   netG:backward(input_ctx, grad)


   local errGtotal = errG * opt.wt_GANg + errG_n * opt.wt_n + errG_l1 * opt.wt_i + errG_s * opt.wt_s
   return errGtotal, gradParametersG
end

---------------------------------------------------------------------------
-- Train Context Encoder
---------------------------------------------------------------------------
-- save opt
paths.mkdir(paths.concat(opt.checkpoints_dir, opt.name))
local file = torch.DiskFile(paths.concat(opt.checkpoints_dir, opt.name, 'opt.txt'), 'w')
file:writeObject(opt)
file:close()

local counter_total = 0
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()
      
      -- load a batch and run G on that batch
      createRealFake()

      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      if opt.use_GAN then  optim.adam(fDx, parametersD, optimStateD) end

      -- (2) Update G network: maximize log(D(G(z)))
      optim.adam(fGx, parametersG, optimStateG)

      -- display
      counter = counter + 1
      if counter % opt.display_iter == 1 and opt.display then
          createRealFake()
          if opt.has_i then
              image.save(paths.concat(opt.checkpoints_dir, opt.name, 'real_i.png'), image.toDisplayTensor(input_full[{{},opt.idx_i,{},{}}]))
              image.save(paths.concat(opt.checkpoints_dir, opt.name, 'real_ctx_i.png'), image.toDisplayTensor(input_ctx[{{},opt.idx_i,{},{}}]))
              if netG.netinfo.out_idx_i then 
                 image.save(paths.concat(opt.checkpoints_dir, opt.name, 'fake_i.png'), image.toDisplayTensor(output_full[{{},opt.outidx_i,{},{}}]))
              end
          end

          if opt.has_d then
             image.save(paths.concat(opt.checkpoints_dir, opt.name,'real_d.png'), image.toDisplayTensor(input_full[{{},opt.idx_d,{},{}}]))
             image.save(paths.concat(opt.checkpoints_dir, opt.name,'real_ctx_d.png'), image.toDisplayTensor(input_ctx[{{},opt.idx_d,{},{}}]))
             image.save(paths.concat(opt.checkpoints_dir, opt.name,'fake_d.png'), image.toDisplayTensor(output_full[{{},opt.outidx_d,{},{}}]))
          end

          if opt.has_p then
             image.save(paths.concat(opt.checkpoints_dir, opt.name,'real_p.png'), image.toDisplayTensor(input_full[{{},opt.idx_p,{},{}}]))
             image.save(paths.concat(opt.checkpoints_dir, opt.name,'real_ctx_p.png'), image.toDisplayTensor(input_ctx[{{},opt.idx_p,{},{}}]))
             if netG.netinfo.out_idx_p then 
                image.save(paths.concat(opt.checkpoints_dir, opt.name,'fake_p.png'), image.toDisplayTensor(output_full[{{},opt.outidx_p,{},{}}]))
             end
          end

          if opt.has_n then
             image.save(paths.concat(opt.checkpoints_dir, opt.name,'real_n.png'), image.toDisplayTensor(input_full[{{},opt.idx_n,{},{}}]))
             image.save(paths.concat(opt.checkpoints_dir, opt.name,'real_ctx_n.png'), image.toDisplayTensor(input_ctx[{{},opt.idx_n,{},{}}]))
             local normal_vis = normal_est:float():view(outputsize[1], outputsize[4], outputsize[3], 3):transpose(2,4);
             image.save(paths.concat(opt.checkpoints_dir, opt.name,'fake_n.png'), image.toDisplayTensor(normal_vis))
             
          end
          if opt.has_s then
             image.save(paths.concat(opt.checkpoints_dir, opt.name,'real_s.png'), image.toDisplayTensor(input_full[{{}, opt.idx_s,{},{}}]))
             image.save(paths.concat(opt.checkpoints_dir, opt.name,'real_ctx_s.png'), image.toDisplayTensor(input_ctx[{{},opt.idx_s,{},{}}]))
             image.save(paths.concat(opt.checkpoints_dir, opt.name,'fake_s.png'), image.toDisplayTensor(segmentation_vis))
          end
      end

      counter_total = counter_total+1

      if opt.inc_learning and opt.numofboxes > 3 then -- train with full view 
         opt.singleObjectTrain = math.random()>0.8
         opt.numofboxes = 4
      end

      if counter_total > 10000 or opt.wt_xyz>1 then 
         opt.wt_xyz = 0.01
      end

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%5d / %5d] Time: %.3f NB %d' 
                 .. '  Err_G_L1: %.3f    Err_s: %.3f  Err_n: %.3f   Err_r: %.3f  Err_xyz: %.3f Err_G: %.4f  Err_D: %.4f '):format(
                 epoch, ((i-1) / opt.batchSize), math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, opt.numofboxes, errG_l1, errG_s, errG_n, errG_r, errG_xyz, 
                 errG or -1, errD or -1))

         print(('errG_nr: %.3f errG_pr: %.3f errG_sr: %.3f errG_xyzr: %.3f'):format(errG_nr, errG_pr, errG_sr, errG_xyzr))
         print(('counter_total: %d  opt.wt_xyz: %.3f errG_distrib: %.3f errG_exist: %.3f'):format(counter_total, opt.wt_xyz, errG_distrib, errG_exist))
         
      end

      -- save latest model
      if counter % opt.save_latest_freq == 1 then
          print(('saving the latest model (epoch %d, iters %d)'):format(epoch, counter))
          torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), netG:clearState())
          torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), netD:clearState())
      end
   end
   
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   if epoch % opt.save_epoch_freq == 0 then
      paths.mkdir(paths.concat(opt.checkpoints_dir, opt.name))
      torch.save(paths.concat(opt.checkpoints_dir, opt.name,  epoch .. '_net_G.t7'), netG:clearState())
      torch.save(paths.concat(opt.checkpoints_dir, opt.name, epoch .. '_net_D.t7'), netD:clearState())
   end
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(epoch, opt.niter, epoch_tm:time().real))
end
