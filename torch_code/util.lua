local util = {}
function util.basename_batch(batch)
    for i = 1, #batch do
        batch[i] = paths.basename(batch[i])
    end
    return batch
end
function util.file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end


function util.load(filename, opt)
    require 'cudnn'
    require 'cunn'
    local net = torch.load(filename)
    if opt.gpu > 0 then
        net:cuda()
        -- calling cuda on cudnn saved nngraphs doesn't change all variables to cuda, so do it below
        if net.forwardnodes then
            for i=1,#net.forwardnodes do
                if net.forwardnodes[i].data.module then
                    net.forwardnodes[i].data.module:cuda()
                end
            end
        end
        
    else
        net:float()
    end
    net:apply(function(m) if m.weight then 
        m.gradWeight = m.weight:clone():zero(); 
        m.gradBias = m.bias:clone():zero(); end end)
    return net
end

-- modules that can be converted to nn seamlessly
local layer_list = {
  'BatchNormalization',
  'SpatialBatchNormalization',
  'SpatialConvolution',
  'SpatialCrossMapLRN',
  'SpatialFullConvolution',
  'SpatialMaxPooling',
  'SpatialAveragePooling',
  'ReLU',
  'Tanh',
  'Sigmoid',
  'SoftMax',
  'LogSoftMax',
  'VolumetricBatchNormalization',
  'VolumetricConvolution',
  'VolumetricFullConvolution',
  'VolumetricMaxPooling',
  'VolumetricAveragePooling',
}

-- goes over a given net and converts all layers to dst backend
-- for example: net = cudnn_convert_custom(net, cudnn)
-- same as cudnn.convert with gModule check commented out
function cudnn_convert_custom(net, dst, exclusion_fn)
  return net:replace(function(x)
    local y = 0
    local src = dst == nn and cudnn or nn
    local src_prefix = src == nn and 'nn.' or 'cudnn.'
    local dst_prefix = dst == nn and 'nn.' or 'cudnn.'

    local function convert(v)
      local y = {}
      torch.setmetatable(y, dst_prefix..v)
      if v == 'ReLU' then y = dst.ReLU() end -- because parameters
      for k,u in pairs(x) do y[k] = u end
      if src == cudnn and x.clearDesc then x.clearDesc(y) end
      if src == cudnn and v == 'SpatialAveragePooling' then
        y.divide = true
        y.count_include_pad = v.mode == 'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING'
      end
      return y
    end

    if exclusion_fn and exclusion_fn(x) then
      return x
    end
    local t = torch.typename(x)
    if t == 'nn.SpatialConvolutionMM' then
      y = convert('SpatialConvolution')
    elseif t == 'inn.SpatialCrossResponseNormalization' then
      y = convert('SpatialCrossMapLRN')
    else
      for i,v in ipairs(layer_list) do
        if torch.typename(x) == src_prefix..v then
          y = convert(v)
        end
      end
    end
    return y == 0 and x or y
  end)
end

function util.cudnn(net)
    require 'cudnn'
    return cudnn_convert_custom(net, cudnn)
end

return util
