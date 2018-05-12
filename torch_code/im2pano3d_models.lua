require 'nn'
require 'cudnn'
require 'cunn'
require 'nngraph'
require 'image'
require 'hdf5'

function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

function defineG(opt)
    -- decide input array
    local input_nc_array,input_start_array
    if opt.Gtype_in == 'rgbpns' then
       input_nc_array = {3,1,3,1}
       in_dim = 3+1+3+1
       input_start_array = {opt.idx_i[1],opt.idx_p[1],opt.idx_n[1],opt.idx_s[1]}
    elseif opt.Gtype_in == 'rgbpn' then 
       input_nc_array = {3,1,3}
       in_dim = 3+1+3
       input_start_array = {opt.idx_i[1],opt.idx_p[1],opt.idx_n[1]}
    elseif opt.Gtype_in == 'rgb' then 
       input_nc_array = {3}
       in_dim = 3
       input_start_array = {opt.idx_i[1]}
    elseif opt.Gtype_in == 'pns' then 
       input_nc_array = {1,3,1}
       in_dim = 1+3+1
       input_start_array = {opt.idx_p[1],opt.idx_n[1],opt.idx_s[1]}
    elseif opt.Gtype_in == 'pn' then 
      input_nc_array = {1,3}
      in_dim = 1+3
      input_start_array = {opt.idx_p[1],opt.idx_n[1]}
    elseif opt.Gtype_in == 's' then 
      input_nc_array = {1}
      in_dim = 1
      input_start_array = {opt.idx_s[1]}
    end

    -- output of the netG
    local out_idx_i,out_idx_p,out_idx_n,out_idx_s,out_idx_xyz,out_idx_rm,out_idx_d
    local out_idx_pr,out_idx_nr,out_idx_sr,out_idx_xyzr,out_idx_att

    local output_nc_array
    local num_output
    if opt.Gtype_out == 'rgbpns' then
       out_idx_i = 1
       out_idx_p = 2
       out_idx_n = 3
       out_idx_s = 4

       num_output = 4
       output_nc_array = {3,1,3,opt.nClasses+1}
       out_dim = 3+1+3+opt.nClasses+1
    elseif opt.Gtype_out == 'pns' then 
       out_idx_p = 1
       out_idx_n = 2
       out_idx_s = 3
       num_output = 3
       output_nc_array = {1,3,opt.nClasses+1}
       out_dim = 1+3+opt.nClasses+1
    elseif opt.Gtype_out == 'rgb' then 
      out_idx_i = -1
      num_output = 1
      out_dim = 3
      output_nc_array = {3}
    elseif opt.Gtype_out == 'd' then 
      out_idx_p = -1
      num_output = 1
      out_dim = 1
    elseif opt.Gtype_out == 'pn' then 
      out_idx_p = 1
      out_idx_n = 2
      num_output = 2
      output_nc_array = {1,3}
      out_dim = 1+3
    elseif opt.Gtype_out == 's' then  
       out_idx_s = 1
       output_nc_array = {opt.nClasses+1}
       num_output = 1
       out_dim = opt.nClasses+1
    end

    if opt.loss_xyz then
       out_idx_xyz = num_output+1
       num_output = num_output+1
    end

    if opt.predroomType then
       out_idx_rm = num_output+1
       num_output = num_output+1
    end

    if opt.attribute_exist then 
       out_idx_att = {num_output+1,num_output+1+opt.nClasses-opt.nClasses_r-1}
       num_output = num_output+(opt.nClasses-opt.nClasses_r)
    end


    local netG = nil 
    if opt.Gtype_in == 'rgbpns' and  opt.Gtype_out == 'pns' then 
        netG = defineG_unet_rgbpns_pns(input_nc_array,input_start_array,output_nc_array, opt.ngf, opt)
    elseif opt.Gtype_in == 'rgbpn' and  opt.Gtype_out == 'pns' then 
        netG = defineG_unet_rgbpn_pns(input_nc_array,input_start_array,output_nc_array, opt.ngf, opt)
    elseif opt.Gtype_in == 'pns' and  opt.Gtype_out == 'pns' then 
        netG = defineG_unet_pns_pns(input_nc_array,input_start_array,output_nc_array, opt.ngf, opt)
    elseif opt.Gtype_in == 'pn' and  opt.Gtype_out == 'pn' then 
        netG = defineG_unet_pn_pn(input_nc_array,input_start_array,output_nc_array, opt.ngf, opt)
    elseif  opt.Gtype_in == 's' and  opt.Gtype_out == 's' then 
        netG = defineG_unet_s_s(input_nc_array,input_start_array, output_nc_array, opt.ngf, opt)
    elseif opt.Gtype_in == 'rgb' and  opt.Gtype_out == 'pns' then 
        netG = defineG_unet_rgb_pns(input_nc_array,input_start_array,output_nc_array, opt.ngf, opt)
    elseif opt.Gtype_in == 'pns' and  opt.Gtype_out == 'rgb' then 
        netG = defineG_unet_pns_rgb(input_nc_array,input_start_array,output_nc_array, opt.ngf, opt)
    else 
        netG = defineG_unet(opt.nc, opt.nc, opt.ngf)
    end

    local netinfo = {}
    netinfo.out_idx_i = out_idx_i
    netinfo.out_idx_p = out_idx_p
    netinfo.out_idx_n = out_idx_n
    netinfo.out_idx_s = out_idx_s
    netinfo.out_idx_d = out_idx_d
    netinfo.out_idx_xyz = out_idx_xyz
    netinfo.out_idx_rm = out_idx_rm
    netinfo.output_nc_array = output_nc_array
    netinfo.out_dim = out_dim
    netinfo.num_output = num_output
    netinfo.in_dim = in_dim


    netG.netinfo = netinfo
    return netG
end
function defineD(opt,netGinfo)
    local netD = nil 
    if opt.Dtype == 'eq' then 
       netD = defineD_n_layers_eq(netGinfo.output_nc_array, opt.ndf, opt.netD_nlayer) 
    elseif opt.Dtype == 'sep' then 
       netD = defineD_n_layers_sep(netGinfo.output_nc_array, opt.ndf, opt.netD_nlayer) 
    else
       netD = defineD_n_layers(netGinfo.out_dim, opt.ndf, opt.netD_nlayer) 
    end
    return netD
end

function defineG_unet_pns_rgb(input_nc, input_start, output_nc, ngf, opt)
    local netG = nil
    local in_0 = - nn.Identity() 
    
    local e_d  = in_0 -nn.Narrow(2,input_start[1],input_nc[1])
    local e_n  = in_0 -nn.Narrow(2,input_start[2],input_nc[2])
    local e_s  = in_0 -nn.Narrow(2,input_start[3],input_nc[3])

    
    local e0_d = e_d - nn.SpatialConvolution(input_nc[1], ngf/2, 3, 3, 1, 1, 1, 1) -- rgbd image
    local e1_d = e0_d - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf/2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local e2_d = e1_d - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)

    -- input is (ngf * 4) x  640 x 256
    local e0_n = e_n - nn.SpatialConvolution(input_nc[2], ngf/2, 3, 3, 1, 1, 1, 1) -- segment image
    local e1_n = e0_n - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf/2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local e2_n = e1_n - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
   
    -- input is (ngf * 4) x  640 x 256
    local e0_s = e_s - nn.SpatialConvolution(input_nc[3], ngf/2, 3, 3, 1, 1, 1, 1) -- segment image
    local e1_s = e0_s - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf/2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local e2_s = e1_s - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)

    local e2 = {e2_d, e2_n, e2_s} - nn.JoinTable(2) 
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2 * 3, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x  80 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x  40 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 20 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    --input is (ngf * 8) x 10 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 5 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 16, 5, 2, 1, 1, 0,0) - nn.SpatialBatchNormalization(ngf * 16)
    -- input is (ngf * 8) x 1 x 1
    if opt.predroomType then 
       i0 =  e8 
       i1 = i0 - nn.View(-1, ngf * 16) - nn.ReLU(true) - nn.Linear(ngf * 16, ngf * 4)
       i2 = i1 - nn.ReLU(true) - nn.Linear(ngf * 4, opt.nrmType)
       opt.nz = 0
    else
       opt.nz = 0
       i0 = e8
    end

    local attribute_predict ={}
    if opt.attribute_exist then 
       for i = 1,(opt.nClasses-opt.nClasses_r) do -- for all the class that not room structure compute attribute 
           attribute_predict[i] = e8 - nn.View(-1, ngf * 16) - nn.ReLU(true) - nn.Linear(ngf * 16, 2)
       end
    end

    ---- Decoder
    local d1_ = i0 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 16 + opt.nz, ngf * 8, 5, 2, 1, 1, 0, 0) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 1
    local d1 = {d1_,e7} - nn.JoinTable(2)

    -- input is (ngf * 8) x 2
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4
    local d2 = {d2_,e6} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8
    local d3 = {d3_,e5} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16
    local d4 = {d4_,e4} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32
    local d5 = {d5_,e3} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
 

    
    -- output for p
    local d6_d  = {d6_,e2_d} - nn.JoinTable(2)
    local d7_d_ = d6_d - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local d7_d  = {d7_d_,e1_d} - nn.JoinTable(2)
    local d8_d  = d7_d - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) 
    local d9_d  = d8_d  - nn.ReLU(true) - nn.SpatialConvolution(ngf, output_nc[1], 3, 3, 1,1, 1, 1) - nn.Tanh() -- n * 1*  640 * 256

    
    
    if opt.predroomType then 
       netG = nn.gModule({in_0},{d9_d, i2}) 
    else
       netG = nn.gModule({in_0},{d9_d})
    end

    netG.opt = opt
    netG.BottleneckLayer = e8
    return netG
end


function defineG_unet_rgb_pns(input_nc, input_start, output_nc, ngf, opt)
    local netG = nil
    local in_0 = - nn.Identity() 
    local e_i  = in_0 -nn.Narrow(2,input_start[1],input_nc[1])
    
    -- input is (ngf * 4) x  640 x 256
    local e0_i = e_i - nn.SpatialConvolution(input_nc[1], ngf/2, 3, 3, 1, 1, 1, 1) -- rgbd image
    local e1_i = e0_i - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf/2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local e2 = e1_i - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x  80 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x  40 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 20 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    --input is (ngf * 8) x 10 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 5 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 16, 5, 2, 1, 1, 0,0) - nn.SpatialBatchNormalization(ngf * 16)
    -- input is (ngf * 8) x 1 x 1
    if opt.predroomType then 
       i0 =  e8 
       i1 = i0 - nn.View(-1, ngf * 16) - nn.ReLU(true) - nn.Linear(ngf * 16, ngf * 4)
       i2 = i1 - nn.ReLU(true) - nn.Linear(ngf * 4, opt.nrmType)
       opt.nz = 0
    else
        opt.nz = 0
        i0 = e8
    end

    ---- Decoder
    local d1_ = i0 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 16 + opt.nz, ngf * 8, 5, 2, 1, 1, 0, 0) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 1
    local d1 = {d1_,e7} - nn.JoinTable(2)

    -- input is (ngf * 8) x 2
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4
    local d2 = {d2_,e6} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8
    local d3 = {d3_,e5} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16
    local d4 = {d4_,e4} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32
    local d5 = {d5_,e3} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    local d6  = {d6_,e2} - nn.JoinTable(2)

    -- output for p
    local d7_d_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local d7_d  = {d7_d_,e1_i} - nn.JoinTable(2)
    local d8_d  = d7_d - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) 
    local d9_d  = d8_d  - nn.ReLU(true) - nn.SpatialConvolution(ngf, output_nc[1], 3, 3, 1,1, 1, 1) - nn.Tanh() -- n * 1*  640 * 256
    
    
    -- output for normal 
    local d7_n_  = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local d7_n   = {d7_n_,e1_i} - nn.JoinTable(2)
    local d8_n   = d7_n - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) 
    local d9_n   = d8_n - nn.ReLU(true) - nn.SpatialConvolution(ngf, output_nc[2], 3, 3, 1,1, 1, 1) -- n * 3*  640 * 256
    local d10_n  = d9_n -nn.Transpose({2,4}) - nn.View(-1,output_nc[2]) - nn.Normalize(2)  
    

    -- output for seg 
    local d7_s_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local d7_s  = {d7_s_,e1_i} - nn.JoinTable(2)
    local d8_s  = d7_s - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) 
    local d9_s  = d8_s - nn.ReLU(true) - nn.SpatialConvolution(ngf, output_nc[3], 3, 3, 1,1, 1, 1)  -- n * 3*  640 * 256

    
    -- loss for the pnmap
    local tw = - nn.Identity() 
    local pm1 =  d9_d - nn.AddConstant(1, false) - nn.MulConstant(4.0959)
    local nm1 =  d10_n - nn.View(-1, opt.fineSize*opt.WtoHRatio, opt.fineSize, 3) - nn.Transpose({2,4})
    local pnnet1 = {tw,nm1} - nn.CMulTable()
    local pnnet2 = pnnet1 - nn.Sum(2) - nn.AddConstant(0.0001, true)
    local pnnet3 = {pm1,pnnet2} - nn.CDivTable()
    local pnnet4 = pnnet3 -nn.Replicate(3,2)
    local pnnet5 = {tw,pnnet4} - nn.CMulTable()-nn.Clamp(-20, 20)

    
    if opt.predroomType then
       netG = nn.gModule({in_0,tw},{d9_d, d10_n, d9_s, pnnet5,  i2}) 
    else
       netG = nn.gModule({in_0,tw},{d9_d, d10_n, d9_s, pnnet5}) 
    end

    netG.opt = opt
    netG.BottleneckLayer = e8
    return netG
end

function defineG_unet_s_s(input_nc, input_start, output_nc, ngf, opt)
    local netG = nil
    local e1 = - nn.SpatialConvolution(input_nc[1], ngf/2, 3, 3, 1, 1, 1, 1) 
    local e1_s = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf/2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local e2 = e1_s - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)

    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x  80 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x  40 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 20 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    --input is (ngf * 8) x 10 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 5 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 16, 5, 2, 1, 1, 0,0) - nn.SpatialBatchNormalization(ngf * 16)
    -- input is (ngf * 8) x 1 x 1
    if opt.predroomType then 
       i0 =  e8 
       i1 = i0 - nn.View(-1, ngf * 16) - nn.ReLU(true) - nn.Linear(ngf * 16, ngf * 4)
       i2 = i1 - nn.ReLU(true) - nn.Linear(ngf * 4, opt.nrmType)
       opt.nz = 0
    else
       i0 = e8
       opt.nz = 0
    end

    local attribute_predict ={}
    if opt.attribute_exist then 
       for i = 1,(opt.nClasses-opt.nClasses_r) do -- for all the class that not room structure compute attribute 
           attribute_predict[i] = e8 - nn.View(-1, ngf * 16) - nn.ReLU(true) - nn.Linear(ngf * 16, 2)
       end
    end

    ---- Decoder
    local d1_ = i0 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 16 + opt.nz, ngf * 8, 5, 2, 1, 1, 0, 0) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 1
    local d1 = {d1_,e7} - nn.JoinTable(2)

    -- input is (ngf * 8) x 2
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4
    local d2 = {d2_,e6} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8
    local d3 = {d3_,e5} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16
    local d4 = {d4_,e4} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32
    local d5 = {d5_,e3} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)

    
    -- output for seg 
    local d6_s  = {d6_,e2} - nn.JoinTable(2)
    local d7_s_ = d6_s - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local d7_s  = {d7_s_,e1_s} - nn.JoinTable(2)
    local d8_s  = d7_s - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) 
    local d9_s  = d8_s - nn.ReLU(true) - nn.SpatialConvolution(ngf, output_nc[1], 3, 3, 1,1, 1, 1)  -- n * 3*  640 * 256


    if opt.predroomType then 
       netG = nn.gModule({e1},{d9_s, i2}) 
    else
       netG = nn.gModule({e1},{d9_s})
    end

    netG.opt = opt
    netG.BottleneckLayer = e8
    return netG
end

function defineG_unet_pns_pns(input_nc, input_start, output_nc, ngf, opt)
    local netG = nil
    local in_0 = - nn.Identity() 
    
    local e_d  = in_0 -nn.Narrow(2,input_start[1],input_nc[1])
    local e_n  = in_0 -nn.Narrow(2,input_start[2],input_nc[2])
    local e_s  = in_0 -nn.Narrow(2,input_start[3],input_nc[3])

    
    local e0_d = e_d - nn.SpatialConvolution(input_nc[1], ngf/2, 3, 3, 1, 1, 1, 1) -- rgbd image
    local e1_d = e0_d - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf/2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local e2_d = e1_d - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)

    -- input is (ngf * 4) x  640 x 256
    local e0_n = e_n - nn.SpatialConvolution(input_nc[2], ngf/2, 3, 3, 1, 1, 1, 1) -- segment image
    local e1_n = e0_n - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf/2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local e2_n = e1_n - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
   
    -- input is (ngf * 4) x  640 x 256
    local e0_s = e_s - nn.SpatialConvolution(input_nc[3], ngf/2, 3, 3, 1, 1, 1, 1) -- segment image
    local e1_s = e0_s - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf/2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local e2_s = e1_s - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)

    local e2 = {e2_d, e2_n, e2_s} - nn.JoinTable(2) 
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2 * 3, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x  80 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x  40 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 20 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    --input is (ngf * 8) x 10 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 5 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 16, 5, 2, 1, 1, 0,0) - nn.SpatialBatchNormalization(ngf * 16)
    -- input is (ngf * 8) x 1 x 1
    if opt.predroomType then 
       i0 =  e8 
       i1 = i0 - nn.View(-1, ngf * 16) - nn.ReLU(true) - nn.Linear(ngf * 16, ngf * 4)
       i2 = i1 - nn.ReLU(true) - nn.Linear(ngf * 4, opt.nrmType)
       opt.nz = 0
    else
       opt.nz = 0
       i0 = e8
    end

    local attribute_predict ={}
    if opt.attribute_exist then 
       for i = 1,(opt.nClasses-opt.nClasses_r) do -- for all the class that not room structure compute attribute 
           attribute_predict[i] = e8 - nn.View(-1, ngf * 16) - nn.ReLU(true) - nn.Linear(ngf * 16, 2)
       end
    end

    ---- Decoder
    local d1_ = i0 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 16 + opt.nz, ngf * 8, 5, 2, 1, 1, 0, 0) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 1
    local d1 = {d1_,e7} - nn.JoinTable(2)

    -- input is (ngf * 8) x 2
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4
    local d2 = {d2_,e6} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8
    local d3 = {d3_,e5} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16
    local d4 = {d4_,e4} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32
    local d5 = {d5_,e3} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
 

    
    -- output for p
    local d6_d  = {d6_,e2_d} - nn.JoinTable(2)
    local d7_d_ = d6_d - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local d7_d  = {d7_d_,e1_d} - nn.JoinTable(2)
    local d8_d  = d7_d - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) 
    local d9_d  = d8_d  - nn.ReLU(true) - nn.SpatialConvolution(ngf, output_nc[1], 3, 3, 1,1, 1, 1) - nn.Tanh() -- n * 1*  640 * 256

    -- output for normal 
    local d6_n   = {d6_,e2_n} - nn.JoinTable(2)
    local d7_n_  = d6_n - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local d7_n   = {d7_n_,e1_n} - nn.JoinTable(2)
    local d8_n   = d7_n - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) 
    local d9_n   = d8_n - nn.ReLU(true) - nn.SpatialConvolution(ngf, output_nc[2], 3, 3, 1,1, 1, 1) -- n * 3*  640 * 256
    local d10_n  = d9_n -nn.Transpose({2,4}) - nn.View(-1,output_nc[2]) - nn.Normalize(2)  
    

    -- output for seg 
    local d6_s  = {d6_,e2_s} - nn.JoinTable(2)
    local d7_s_ = d6_s - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local d7_s  = {d7_s_,e1_s} - nn.JoinTable(2)
    local d8_s  = d7_s - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) 
    local d9_s  = d8_s - nn.ReLU(true) - nn.SpatialConvolution(ngf, output_nc[3], 3, 3, 1,1, 1, 1)  -- n * 3*  640 * 256

    -- loss for the pnmap
    local tw = - nn.Identity() 
    local pm1 =  d9_d - nn.AddConstant(1, false) - nn.MulConstant(4.0959)
    local nm1 =  d10_n - nn.View(-1, opt.fineSize*opt.WtoHRatio, opt.fineSize, 3) - nn.Transpose({2,4})
    local pnnet1 = {tw,nm1} - nn.CMulTable()
    local pnnet2 = pnnet1 - nn.Sum(2) - nn.AddConstant(0.0001, true)
    local pnnet3 = {pm1,pnnet2} - nn.CDivTable()
    local pnnet4 = pnnet3 -nn.Replicate(3,2)
    local pnnet5 = {tw,pnnet4} - nn.CMulTable()-nn.Clamp(-20, 20)
    
    if opt.attribute_exist then
       local out = {d9_d, d10_n, d9_s,  pnnet5, i2}
       for i = 1,(opt.nClasses-opt.nClasses_r) do -- for all the class that not room structure compute attribute 
           out[5+i] = attribute_predict[i] 
       end
       netG = nn.gModule({in_0,tw},out) 
    elseif opt.predroomType then 
       netG = nn.gModule({in_0,tw},{d9_d, d10_n, d9_s,  pnnet5, i2}) 
    else
       netG = nn.gModule({in_0,tw},{d9_d, d10_n, d9_s, pnnet5})
    end

    netG.opt = opt
    netG.BottleneckLayer = e8
    return netG
end

function defineG_unet_pn_pn(input_nc, input_start, output_nc, ngf, opt)
    local netG = nil
    local in_0 = - nn.Identity() 
    
    local e_d  = in_0 -nn.Narrow(2,input_start[1],input_nc[1])
    local e_n  = in_0 -nn.Narrow(2,input_start[2],input_nc[2])
    
    
    local e0_d = e_d - nn.SpatialConvolution(input_nc[1], ngf/2, 3, 3, 1, 1, 1, 1) -- rgbd image
    local e1_d = e0_d - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf/2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local e2_d = e1_d - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)

    -- input is (ngf * 4) x  640 x 256
    local e0_n = e_n - nn.SpatialConvolution(input_nc[2], ngf/2, 3, 3, 1, 1, 1, 1) -- segment image
    local e1_n = e0_n - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf/2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local e2_n = e1_n - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
   
   
    local e2 = {e2_d, e2_n} - nn.JoinTable(2) 
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x  80 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x  40 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 20 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    --input is (ngf * 8) x 10 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 5 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 16, 5, 2, 1, 1, 0,0) - nn.SpatialBatchNormalization(ngf * 16)
    -- input is (ngf * 8) x 1 x 1
    if opt.predroomType then 
       i0 =  e8 
       i1 = i0 - nn.View(-1, ngf * 16) - nn.ReLU(true) - nn.Linear(ngf * 16, ngf * 4)
       i2 = i1 - nn.ReLU(true) - nn.Linear(ngf * 4, opt.nrmType)
       opt.nz = 0
    else
       opt.nz = 0
       i0 = e8
    end

    local attribute_predict ={}
    if opt.attribute_exist then 
       for i = 1,(opt.nClasses-opt.nClasses_r) do -- for all the class that not room structure compute attribute 
           attribute_predict[i] = e8 - nn.View(-1, ngf * 16) - nn.ReLU(true) - nn.Linear(ngf * 16, 2)
       end
    end

    ---- Decoder
    local d1_ = i0 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 16 + opt.nz, ngf * 8, 5, 2, 1, 1, 0, 0) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 1
    local d1 = {d1_,e7} - nn.JoinTable(2)

    -- input is (ngf * 8) x 2
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4
    local d2 = {d2_,e6} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8
    local d3 = {d3_,e5} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16
    local d4 = {d4_,e4} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32
    local d5 = {d5_,e3} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
 

    
    -- output for p
    local d6_d  = {d6_,e2_d} - nn.JoinTable(2)
    local d7_d_ = d6_d - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local d7_d  = {d7_d_,e1_d} - nn.JoinTable(2)
    local d8_d  = d7_d - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) 
    local d9_d  = d8_d  - nn.ReLU(true) - nn.SpatialConvolution(ngf, output_nc[1], 3, 3, 1,1, 1, 1) - nn.Tanh() -- n * 1*  640 * 256

    -- output for normal 
    local d6_n   = {d6_,e2_n} - nn.JoinTable(2)
    local d7_n_  = d6_n - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local d7_n   = {d7_n_,e1_n} - nn.JoinTable(2)
    local d8_n   = d7_n - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) 
    local d9_n   = d8_n - nn.ReLU(true) - nn.SpatialConvolution(ngf, output_nc[2], 3, 3, 1,1, 1, 1) -- n * 3*  640 * 256
    local d10_n  = d9_n -nn.Transpose({2,4}) - nn.View(-1,output_nc[2]) - nn.Normalize(2)  
    

   
    -- loss for the pnmap
    local tw = - nn.Identity() 
    local pm1 =  d9_d - nn.AddConstant(1, false) - nn.MulConstant(4.0959)
    local nm1 =  d10_n - nn.View(-1, opt.fineSize*opt.WtoHRatio, opt.fineSize, 3) - nn.Transpose({2,4})
    local pnnet1 = {tw,nm1} - nn.CMulTable()
    local pnnet2 = pnnet1 - nn.Sum(2) - nn.AddConstant(0.0001, true)
    local pnnet3 = {pm1,pnnet2} - nn.CDivTable()
    local pnnet4 = pnnet3 -nn.Replicate(3,2)
    local pnnet5 = {tw,pnnet4} - nn.CMulTable()-nn.Clamp(-20, 20)
    
    if opt.predroomType then 
       netG = nn.gModule({in_0,tw},{d9_d, d10_n,  pnnet5, i2}) 
    else
       netG = nn.gModule({in_0,tw},{d9_d, d10_n, pnnet5})
    end

    netG.opt = opt
    netG.BottleneckLayer = e8
    return netG
end

function defineG_unet_rgbpn_pns(input_nc, input_start, output_nc, ngf, opt)
    local netG = nil
    local in_0 = - nn.Identity() 
    local e_i  = in_0 -nn.Narrow(2,input_start[1],input_nc[1])
    local e_d  = in_0 -nn.Narrow(2,input_start[2],input_nc[2])
    local e_n  = in_0 -nn.Narrow(2,input_start[3],input_nc[3])

    -- input is (ngf * 4) x  640 x 256
    local e0_i = e_i - nn.SpatialConvolution(input_nc[1], ngf/2, 3, 3, 1, 1, 1, 1) -- rgbd image
    local e1_i = e0_i - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf/2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local e2_i = e1_i - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    
    local e0_d = e_d - nn.SpatialConvolution(input_nc[2], ngf/2, 3, 3, 1, 1, 1, 1) -- rgbd image
    local e1_d = e0_d - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf/2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local e2_d = e1_d - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)

    -- input is (ngf * 4) x  640 x 256
    local e0_n = e_n - nn.SpatialConvolution(input_nc[3], ngf/2, 3, 3, 1, 1, 1, 1) -- segment image
    local e1_n = e0_n - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf/2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local e2_n = e1_n - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
   
    
    local e2 = {e2_i, e2_d,e2_n} - nn.JoinTable(2) 
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2 * 3, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x  80 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x  40 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 20 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    --input is (ngf * 8) x 10 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 5 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 16, 5, 2, 1, 1, 0,0) - nn.SpatialBatchNormalization(ngf * 16)
    -- input is (ngf * 8) x 1 x 1
    if opt.predroomType then 
       i0 =  e8 
       i1 = i0 - nn.View(-1, ngf * 16) - nn.ReLU(true) - nn.Linear(ngf * 16, ngf * 4)
       i2 = i1 - nn.ReLU(true) - nn.Linear(ngf * 4, opt.nrmType)
       opt.nz = 0
    else
        opt.nz = 0
        i0 = e8
    end

    ---- Decoder
    local d1_ = i0 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 16 + opt.nz, ngf * 8, 5, 2, 1, 1, 0, 0) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 1
    local d1 = {d1_,e7} - nn.JoinTable(2)

    -- input is (ngf * 8) x 2
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4
    local d2 = {d2_,e6} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8
    local d3 = {d3_,e5} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16
    local d4 = {d4_,e4} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32
    local d5 = {d5_,e3} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
 

    
    -- output for p
    local d6_d  = {d6_,e2_d} - nn.JoinTable(2)
    local d7_d_ = d6_d - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local d7_d  = {d7_d_,e1_d} - nn.JoinTable(2)
    local d8_d  = d7_d - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) 
    local d9_d  = d8_d  - nn.ReLU(true) - nn.SpatialConvolution(ngf, output_nc[1], 3, 3, 1,1, 1, 1) - nn.Tanh() -- n * 1*  640 * 256

    -- output for normal 
    local d6_n   = {d6_,e2_n} - nn.JoinTable(2)
    local d7_n_  = d6_n - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local d7_n   = {d7_n_,e1_n} - nn.JoinTable(2)
    local d8_n   = d7_n - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) 
    local d9_n   = d8_n - nn.ReLU(true) - nn.SpatialConvolution(ngf, output_nc[2], 3, 3, 1,1, 1, 1) -- n * 3*  640 * 256
    local d10_n  = d9_n -nn.Transpose({2,4}) - nn.View(-1,output_nc[2]) - nn.Normalize(2)  
    

    -- output for seg 
    local d7_s_ = d6_   - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local d8_s  = d7_s_ - nn.ReLU(true) - nn.SpatialFullConvolution(ngf, ngf, 4, 4, 2, 2, 1, 1) 
    local d9_s  = d8_s  - nn.ReLU(true) - nn.SpatialConvolution(ngf, output_nc[3], 3, 3, 1,1, 1, 1)  -- n * 3*  640 * 256

    -- loss for the pnmap
    local tw = - nn.Identity() 
    local pm1 =  d9_d - nn.AddConstant(1, false) - nn.MulConstant(4.0959)
    local nm1 =  d10_n - nn.View(-1, opt.fineSize*opt.WtoHRatio, opt.fineSize, 3) - nn.Transpose({2,4})
    local pnnet1 = {tw,nm1} - nn.CMulTable()
    local pnnet2 = pnnet1 - nn.Sum(2) - nn.AddConstant(0.0001, true)
    local pnnet3 = {pm1,pnnet2} - nn.CDivTable()
    local pnnet4 = pnnet3 -nn.Replicate(3,2)
    local pnnet5 = {tw,pnnet4} - nn.CMulTable()-nn.Clamp(-20, 20)
    
    if opt.predroomType then
       netG = nn.gModule({in_0,tw},{d9_d, d10_n, d9_s,  pnnet5, i2}) 
    else
       netG = nn.gModule({in_0,tw},{d9_d, d10_n, d9_s, pnnet5})
    end

    netG.opt = opt
    netG.BottleneckLayer = e8
    return netG
end


function defineG_unet_rgbpns_pns(input_nc, input_start, output_nc, ngf, opt)
    local netG = nil
    local in_0 = - nn.Identity() 
    local e_i  = in_0 -nn.Narrow(2,input_start[1],input_nc[1])
    local e_d  = in_0 -nn.Narrow(2,input_start[2],input_nc[2])
    local e_n  = in_0 -nn.Narrow(2,input_start[3],input_nc[3])
    local e_s  = in_0 -nn.Narrow(2,input_start[4],input_nc[4])

    -- input is (ngf * 4) x  640 x 256
    local e0_i = e_i - nn.SpatialConvolution(input_nc[1], ngf/2, 3, 3, 1, 1, 1, 1) -- rgbd image
    local e1_i = e0_i - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf/2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local e2_i = e1_i - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    
    local e0_d = e_d - nn.SpatialConvolution(input_nc[2], ngf/2, 3, 3, 1, 1, 1, 1) -- rgbd image
    local e1_d = e0_d - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf/2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local e2_d = e1_d - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)

    -- input is (ngf * 4) x  640 x 256
    local e0_n = e_n - nn.SpatialConvolution(input_nc[3], ngf/2, 3, 3, 1, 1, 1, 1) -- segment image
    local e1_n = e0_n - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf/2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local e2_n = e1_n - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
   
    -- input is (ngf * 4) x  640 x 256
    local e0_s = e_s - nn.SpatialConvolution(input_nc[4], ngf/2, 3, 3, 1, 1, 1, 1) -- segment image
    local e1_s = e0_s - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf/2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local e2_s = e1_s - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)

    local e2 = {e2_i, e2_d, e2_n, e2_s} - nn.JoinTable(2) 
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2 * 4, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x  80 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x  40 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 20 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    --input is (ngf * 8) x 10 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 5 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 16, 5, 2, 1, 1, 0,0) - nn.SpatialBatchNormalization(ngf * 16)
    -- input is (ngf * 8) x 1 x 1
    if opt.predroomType then 
       i0 =  e8 
       i1 = i0 - nn.View(-1, ngf * 16) - nn.ReLU(true) - nn.Linear(ngf * 16, ngf * 4)
       i2 = i1 - nn.ReLU(true) - nn.Linear(ngf * 4, opt.nrmType)
       opt.nz = 0
    else
       opt.nz = 0
       i0 = e8
    end

    local attribute_predict ={}
    if opt.attribute_exist then 
       for i = 1,(opt.nClasses-opt.nClasses_r) do -- for all the class that not room structure compute attribute 
           attribute_predict[i] = e8 - nn.View(-1, ngf * 16) - nn.ReLU(true) - nn.Linear(ngf * 16, 2)
       end
    end

    ---- Decoder
    local d1_ = i0 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 16 + opt.nz, ngf * 8, 5, 2, 1, 1, 0, 0) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 1
    local d1 = {d1_,e7} - nn.JoinTable(2)

    -- input is (ngf * 8) x 2
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4
    local d2 = {d2_,e6} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8
    local d3 = {d3_,e5} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16
    local d4 = {d4_,e4} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32
    local d5 = {d5_,e3} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
 

    
    -- output for p
    local d6_d  = {d6_,e2_d} - nn.JoinTable(2)
    local d7_d_ = d6_d - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local d7_d  = {d7_d_,e1_d} - nn.JoinTable(2)
    local d8_d  = d7_d - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) 
    local d9_d  = d8_d  - nn.ReLU(true) - nn.SpatialConvolution(ngf, output_nc[1], 3, 3, 1,1, 1, 1) - nn.Tanh() -- n * 1*  640 * 256

    -- output for normal 
    local d6_n   = {d6_,e2_n} - nn.JoinTable(2)
    local d7_n_  = d6_n - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local d7_n   = {d7_n_,e1_n} - nn.JoinTable(2)
    local d8_n   = d7_n - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) 
    local d9_n   = d8_n - nn.ReLU(true) - nn.SpatialConvolution(ngf, output_nc[2], 3, 3, 1,1, 1, 1) -- n * 3*  640 * 256
    local d10_n  = d9_n -nn.Transpose({2,4}) - nn.View(-1,output_nc[2]) - nn.Normalize(2)  
    

    -- output for seg 
    local d6_s  = {d6_,e2_s} - nn.JoinTable(2)
    local d7_s_ = d6_s - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local d7_s  = {d7_s_,e1_s} - nn.JoinTable(2)
    local d8_s  = d7_s - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) 
    local d9_s  = d8_s - nn.ReLU(true) - nn.SpatialConvolution(ngf, output_nc[3], 3, 3, 1,1, 1, 1)  -- n * 3*  640 * 256

    -- loss for the pnmap
    local tw = - nn.Identity() 
    local pm1 =  d9_d - nn.AddConstant(1, false) - nn.MulConstant(4.0959)
    local nm1 =  d10_n - nn.View(-1, opt.fineSize*opt.WtoHRatio, opt.fineSize, 3) - nn.Transpose({2,4})
    local pnnet1 = {tw,nm1} - nn.CMulTable()
    local pnnet2 = pnnet1 - nn.Sum(2) - nn.AddConstant(0.0001, true)
    local pnnet3 = {pm1,pnnet2} - nn.CDivTable()
    local pnnet4 = pnnet3 -nn.Replicate(3,2)
    local pnnet5 = {tw,pnnet4} - nn.CMulTable()-nn.Clamp(-20, 20)
    
    if opt.attribute_exist then
       local out = {d9_d, d10_n, d9_s,  pnnet5, i2}
       for i = 1,(opt.nClasses-opt.nClasses_r) do -- for all the class that not room structure compute attribute 
           out[5+i] = attribute_predict[i] 
       end
       netG = nn.gModule({in_0,tw},out) 
    elseif opt.predroomType then 
       netG = nn.gModule({in_0,tw},{d9_d, d10_n, d9_s,  pnnet5, i2}) 
    else
       netG = nn.gModule({in_0,tw},{d9_d, d10_n, d9_s, pnnet5})
    end

    netG.opt = opt
    netG.BottleneckLayer = e8
    return netG
end


function defineG_unet(input_nc, output_nc, ngf)
    local netG = nil
    -- input is (nc) x 640 x 256
    local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 320 x 128
    local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x  160 x  64
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x  80 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x  40 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 20 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    --input is (ngf * 8) x 10 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 5 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 5, 2, 1, 1, 0,0) - nn.SpatialBatchNormalization(ngf * 8)
    -- -- input is (ngf * 8) x 1 x 1
    
    ---- Decoder

    local d1_ = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 5, 2, 1, 1, 0, 0) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    local d1 = {d1_,e7} - nn.JoinTable(2)
    -- input is (ngf * 8) x 1
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 2
    local d2 = {d2_,e6} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4
    local d3 = {d3_,e5} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 8
    local d4 = {d4_,e4} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 16
    local d5 = {d5_,e3} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 32
    local d6 = {d6_,e2} - nn.JoinTable(2)
    local d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    -- input is (ngf) x 64
    local d7 = {d7_,e1} - nn.JoinTable(2)
    local d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
    -- input is (nc) x 128
    
    local o1 = d8 - nn.Tanh()
    
    netG = nn.gModule({e1},{o1})

    
    return netG
end

---- 
function defineG_unet_rgbpns_rgbpns(input_nc, input_start, output_nc, ngf, opt)

    local netG = nil
    local in_0 = - nn.Identity() 
    local e_i  = in_0 -nn.Narrow(2,input_start[1],input_nc[1])
    local e_d  = in_0 -nn.Narrow(2,input_start[2],input_nc[2])
    local e_n  = in_0 -nn.Narrow(2,input_start[3],input_nc[3])
    local e_s  = in_0 -nn.Narrow(2,input_start[4],input_nc[4])

    -- input is (ngf * 4) x  640 x 256
    local e0_i = e_i - nn.SpatialConvolution(input_nc[1], ngf/2, 3, 3, 1, 1, 1, 1) -- rgbd image
    local e1_i = e0_i - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf/2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local e2_i = e1_i - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    
    local e0_d = e_d - nn.SpatialConvolution(input_nc[2], ngf/2, 3, 3, 1, 1, 1, 1) -- rgbd image
    local e1_d = e0_d - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf/2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local e2_d = e1_d - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)

    -- input is (ngf * 4) x  640 x 256
    local e0_n = e_n - nn.SpatialConvolution(input_nc[3], ngf/2, 3, 3, 1, 1, 1, 1) -- segment image
    local e1_n = e0_n - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf/2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local e2_n = e1_n - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
   
    -- input is (ngf * 4) x  640 x 256
    local e0_s = e_s - nn.SpatialConvolution(input_nc[4], ngf/2, 3, 3, 1, 1, 1, 1) -- segment image
    local e1_s = e0_s - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf/2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local e2_s = e1_s - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)

    local e2 = {e2_i, e2_d, e2_n, e2_s} - nn.JoinTable(2) 
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2 * 4, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x  80 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x  40 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 20 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    --input is (ngf * 8) x 10 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 5 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 16, 5, 2, 1, 1, 0,0) - nn.SpatialBatchNormalization(ngf * 16)
    -- input is (ngf * 8) x 1 x 1
    if opt.predroomType then 
       i0 =  e8 
       i1 = i0 - nn.View(-1, ngf * 16)-nn.ReLU(true) - nn.Linear(ngf * 16, ngf * 4)
       i2 = i1 - nn.ReLU(true) - nn.Linear(ngf * 4, opt.nrmType)
       opt.nz = 0
    else
        opt.nz = 0
        i0 = e8
    end

    ---- Decoder
    local d1_ = i0 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 16 + opt.nz, ngf * 8, 5, 2, 1, 1, 0, 0) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 1
    local d1 = {d1_,e7} - nn.JoinTable(2)

    -- input is (ngf * 8) x 2
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4
    local d2 = {d2_,e6} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8
    local d3 = {d3_,e5} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16
    local d4 = {d4_,e4} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32
    local d5 = {d5_,e3} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
 

    
    -- output for rgb
    -- input is (ngf * 8) x 16
    local d6_i = {d6_,e2_i} - nn.JoinTable(2)
    local d7_i_ = d6_i - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local d7_i = {d7_i_,e1_i} - nn.JoinTable(2)
    local d8_i  = d7_i - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) 
    local d9_i  = d8_i  - nn.ReLU(true) - nn.SpatialConvolution(ngf, output_nc[1], 3, 3, 1,1, 1, 1) - nn.Tanh() -- n * 3*  640 * 256

    -- output for p
    local d6_d  = {d6_,e2_d} - nn.JoinTable(2)
    local d7_d_ = d6_d - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local d7_d  = {d7_d_,e1_d} - nn.JoinTable(2)
    local d8_d  = d7_d - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) 
    local d9_d  = d8_d  - nn.ReLU(true) - nn.SpatialConvolution(ngf, output_nc[2], 3, 3, 1,1, 1, 1) - nn.Tanh() -- n * 1*  640 * 256

    -- output for normal 
    local d6_n   = {d6_,e2_n} - nn.JoinTable(2)
    local d7_n_  = d6_n - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local d7_n   = {d7_n_,e1_n} - nn.JoinTable(2)
    local d8_n   = d7_n - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) 
    local d9_n   = d8_n - nn.ReLU(true) - nn.SpatialConvolution(ngf, output_nc[3], 3, 3, 1,1, 1, 1) -- n * 3*  640 * 256
    local d10_n  = d9_n -nn.Transpose({2,4}) - nn.View(-1,output_nc[3]) - nn.Normalize(2)  
    

    -- output for seg 
    local d6_s  = {d6_,e2_s} - nn.JoinTable(2)
    local d7_s_ = d6_s - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    local d7_s  = {d7_s_,e1_s} - nn.JoinTable(2)
    local d8_s  = d7_s - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) 
    local d9_s  = d8_s - nn.ReLU(true) - nn.SpatialConvolution(ngf, output_nc[4], 3, 3, 1,1, 1, 1)  -- n * 3*  640 * 256

    -- loss for the pnmap
    local tw = - nn.Identity() 
    local pm1 =  d9_d - nn.AddConstant(1, false) - nn.MulConstant(4.0959)
    local nm1 =  d10_n - nn.View(-1, opt.fineSize*opt.WtoHRatio, opt.fineSize, 3) - nn.Transpose({2,4})
    local pnnet1 = {tw,nm1} - nn.CMulTable()
    local pnnet2 = pnnet1 - nn.Sum(2) - nn.AddConstant(0.0001, true)
    local pnnet3 = {pm1,pnnet2} - nn.CDivTable()
    local pnnet4 = pnnet3 -nn.Replicate(3,2)
    local pnnet5 = {tw,pnnet4} - nn.CMulTable()-nn.Clamp(-20, 20)
    
    if opt.predroomType then
       netG = nn.gModule({in_0,tw},{d9_i, d9_d, d10_n, d9_s,  pnnet5, i2}) 
    else
       netG = nn.gModule({in_0,tw},{d9_i, d9_d, d10_n, d9_s, pnnet5})
    end

    netG.opt = opt
    netG.BottleneckLayer = e8
    return netG
end

function define_distributionNet(opt)
    local constant = 0.00001
    local netDis   = nil
    local seg_est = - nn.Identity()  
    local seg_maxconf = seg_est - nn.Max(2) - nn.Replicate(opt.nClasses+1,2)
    local diff = {seg_est,seg_maxconf}-nn.CSubTable()-nn.AddConstant(constant)
    local maxIndex = diff - nn.Clamp(0, constant) - nn.MulConstant(1/constant)
    local distribution = maxIndex -nn.View(-1,opt.nClasses+1,opt.fineSize*opt.WtoHRatio*opt.fineSize)-nn.Sum(3)
    local distributionSum = distribution - nn.Sum(2) - nn.Replicate(opt.nClasses+1,2)
    local distributionN = {distribution,distributionSum} - nn.CDivTable()
    local netDis = nn.gModule({seg_est},{distributionN})
    return netDis
end

function define_existenceNet(opt)
    -- use mask as input 
    -- use threshold and clamp 
    local constant = 0.00001
    local netExist   = nil
    local seg_est = - nn.Identity()  
    local pred_mask = - nn.Identity() 
    local pred_mask_rep = pred_mask - nn.View(-1,1,opt.fineSize*opt.WtoHRatio*opt.fineSize) - nn.Replicate(opt.nClasses+1,2)
    local seg_maxconf = seg_est - nn.Max(2) - nn.Replicate(opt.nClasses+1,2)
    local diff = {seg_est,seg_maxconf}-nn.CSubTable()-nn.AddConstant(constant)
    local maxIndex = diff - nn.Clamp(0, constant) - nn.MulConstant(1/constant)
    
    local maxIndex_mask = {maxIndex,pred_mask_rep} - nn.CMulTable() 
    
    local pixel_count = maxIndex_mask -nn.View(-1,opt.nClasses+1,opt.fineSize*opt.WtoHRatio*opt.fineSize)-nn.Sum(3)
    local pixel_count_thre = pixel_count - nn.AddConstant(-500)
    local pixel_count_clamp = pixel_count_thre - nn.Clamp(0, 1)
    local netExist = nn.gModule({seg_est,pred_mask},{pixel_count_clamp})
    return netExist
end


function define_PNNet()
    -- get points3dw thorugh forawd path
    local netPN = nil
    local pm = - nn.Identity() 
    local nm = - nn.Identity() 
    local tw = - nn.Identity() 

    local pm1 =  pm - nn.AddConstant(1, false)- nn.MulConstant(4.0959)
    local pnnet1 = {tw,nm} - nn.CMulTable()
    local pnnet2 = pnnet1 - nn.Sum(2)  - nn.AddConstant(0.0001, true)
    local pnnet3 = {pm1,pnnet2} - nn.CDivTable()
    local pnnet4 = pnnet3 -nn.Replicate(3,2)
    local pnnet5 = {tw,pnnet4} - nn.CMulTable()-nn.Clamp(-20, 20)

    netPN = nn.gModule({pm,nm,tw},{pnnet5})
    return netPN;
end

function defineD_n_layers_eq(input_nc, ndf, n_layers) 
    local startId = 1 
    local netD_in = nn.ConcatTable()
    local numofinput = 0
    for i, v in ipairs(input_nc) do
        local netD_i  = nn.Sequential()
        netD_i:add(nn.Narrow(2,startId,input_nc[i]))
        startId = startId+input_nc[i]
        netD_i:add(nn.SpatialConvolution(input_nc[i], ndf/4, 4, 4, 2, 2, 1, 1))
        netD_i:add(nn.LeakyReLU(0.2, true))
        netD_in:add(netD_i)
        numofinput = numofinput+1
    end
    local netD  = nn.Sequential()
    netD:add(netD_in)
    netD:add(nn.JoinTable(2))
    netD:add(defineD_n_layers(numofinput*ndf/4,ndf,n_layers))
    return netD
end


function defineD_n_layers_sep(input_nc, ndf, n_layers) 
    local netD = nn.ConcatTable()
    local netD_i  = nn.Sequential()
    netD_i:add(nn.Narrow(2,1,input_nc[1]))
    netD_i:add(defineD_n_layers(input_nc[1],ndf/4,n_layers))

    local netD_d  = nn.Sequential()
    netD_d:add(nn.Narrow(2,4,input_nc[2]))
    netD_d:add(defineD_n_layers(input_nc[2],ndf/4,n_layers))

    local netD_n = nn.Sequential() 
    netD_n:add(nn.Narrow(2,5,input_nc[3]))
    netD_n:add(defineD_n_layers(input_nc[3],ndf/4,n_layers))
    
    local netD_s = nn.Sequential() 
    netD_s:add(nn.Narrow(2,8,input_nc[4]))
    netD_s:add(defineD_n_layers(input_nc[4],ndf/4,n_layers))

    netD:add(netD_i)
    netD:add(netD_d)
    netD:add(netD_n)
    netD:add(netD_s)

    return netD
end
-- if n=0, then use pixelGAN (rf=1) never use 
-- else rf is 16 if n=1
--            34 if n=2
--            70 if n=3
--            142 if n=4
--            286 if n=5
--            574 if n=6
function defineD_n_layers(input_nc, ndf, n_layers) 
    local netD = nn.Sequential()

    -- input is (nc) x 256 x 256
    netD:add(nn.SpatialConvolution(input_nc, ndf, 4, 4, 2, 2, 1, 1))
    netD:add(nn.LeakyReLU(0.2, true))
    
    local nf_mult = 1
    local nf_mult_prev = 1
    for n = 1, n_layers-1 do 
        nf_mult_prev = nf_mult
        nf_mult = math.min(2^n,8)
        netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, 4, 4, 2, 2, 1, 1))
        netD:add(nn.SpatialBatchNormalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2, true))
    end
    
    -- state size: (ndf*M) x N x N
    nf_mult_prev = nf_mult
    nf_mult = math.min(2^n_layers,8)
    netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, 4, 4, 1, 1, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2, true))
    
    -- state size: (ndf*M*2) x (N-1) x (N-1)
    netD:add(nn.SpatialConvolution(ndf * nf_mult, 1, 4, 4, 1, 1, 1, 1))
    -- state size: 1 x (N-2) x (N-2)
    
    netD:add(nn.Sigmoid())
    -- state size: 1 x (N-2) x (N-2)
    
    return netD
end






function defineD_basic(nc, ndf)

	local netD = nn.Sequential()
	-- input is (nc) x 128 x 320, going into a convolution
	netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
	netD:add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf) x 64 x 160
	netD:add(SpatialConvolution(ndf, ndf, 4, 4, 2, 2, 1, 1))
	netD:add(SpatialBatchNormalization(ndf)):add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf) x 32 x 80

	netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
	netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf*2) x 16 x 40
	netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
	netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf*4) x 8 x 20
	netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
	netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf*8) x 4 x 10
	netD:add(SpatialConvolution(ndf * 8, 1, 10, 4))
	netD:add(nn.Sigmoid())
	-- state size: 1 x 1 x 1
	netD:add(nn.View(1):setNumInputDims(3))
	-- state size: 1
	return netD
end