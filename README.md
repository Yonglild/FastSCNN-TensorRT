# FastSCNN-TensorRT
Implementation of FastSCNN with **TensorRT7** network definition API

FastSCNN:https://github.com/Tramac/Fast-SCNN-pytorch



### How to Run

1. generate .wts

   ```python
   python gen_wts.py 
   ```

2. cmake and make

   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

3. serialize model to plan file

   ```bash
   ./fastscnn -s
   ```

4. deserialize plan file and run inference

   ```bash
   ./fastscnn -d
   ```

### Result

trt result:

![0_false_color_map](resultImg/0_false_color_map.png)

pytorch result:

![](resultImg/frankfurt_000001_058914_leftImg8bitfs1.png)

