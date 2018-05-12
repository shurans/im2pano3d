### Supported input output modalities:
input   | output    | load type 
pns     | pns       | pns
rgbpn   | pns       | rgbpns
rgbpns  | pns       | rgbpns
rgb     | pns       | rgbpns

### Supported mask types:
twoview: Input two views of the panorama and predicts the other two. 
random: random mask 
1camera
3camera:
middlecamera: Input six RGB-D camera looking horizantoally forward 
upcamera: Input six RGB-D camera looking upwards (a)
nomask: No mask applied in the input image. Example usage, input rgb panorama output pns.
mask_1camera_pns: 


[camera config image]

### Organization
The code and data is organized as follows:

### Download
0. Dwonload training and testing panorama data: 
0. Dwonload training and testing panorama data: 


### Training Example 

Export path
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cudnn/v5/lib64
```



Train on SUNCG data

```
cd torch_code/
name=suncg maskType=twoview   loss_xyz=1 loadOpt=rgbpns  Gtype_in=rgbpns   Gtype_out=pns  dataset=suncg  dataPath=../dataset/suncgpano/ DATA_ROOT=../datalist/trainlist_suncgroom8_10000.txt gpu=1 th train.lua  2>&1 | tee ./checkpoints/suncg_baselinefix_pns.log
```

Train on Matterport3D data
```
cd torch_code/
name=mp maskType=twoview   loss_xyz=1 loadOpt=rgbpns  Gtype_in=rgbpns   Gtype_out=pns  dataset=mp  dataPath=../dataset/mpv3/  DATA_ROOT=../datalist/trainlist_mp3.txt gpu=1 th train.lua  2>&1 | tee ./checkpoints/mp_baselinefix_pns.log
```


### Testing 

```
cd torch_code/
name=train1_twoview_rgbpns  maskType=twoview dataset=suncg dataPath=../dataset/suncgpano/  DATA_ROOT=../datalist/testlist_suncgroom8.txt how_many=480 th test.lua
```


### Quick Demo
```
cd torch_code/
name=train1_twoview_rgbpns  maskType=twoview dataset=suncg dataPath=../dataset/mpv3/  DATA_ROOT=../datalist/testlist_suncgroom8.txt how_many=480 th demo.lua
```



### Data Details 

Panorama data representation 


Depth encoding convertion example im Matlab 



