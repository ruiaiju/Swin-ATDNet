
### Training Steps
#### a、Training with the Provided Dataset
1、Place the provided dataset into the dataset folder (do NOT run voc_annotation.py). Modify the file paths in your code accordingly. 
2、Configure parameters in train.py. The default settings are already aligned with the dataset requirements. 
3、Run train.py to start training。  

#### b、Training with a Custom Dataset
1、This repository uses the VOC format for training.  
2、Place your label files (annotations) in dataset/data/Segmentation.    
3、Place your image files in dataset/data/JPEGImages.  
4、Generate the required .txt files by running voc_annotation.py before training.  
5、In train.py, select your backbone model (mobilenet or xception) and downsample factor (8 or 16). Ensure the pre-trained model matches your chosen backbone.    
6、Run train.py to start training.
 


#### Using Custom Trained Weights
1、Follow the training steps above.    
2、In deeplab.py, update the following parameters to match your trained model
3、Run predict.py and input the image path for prediction
4、Configure predict.py for FPS testing, folder-wide testing
   

### Evaluation Steps
Run get_miou.py to calculate the mIoU (mean Intersection over Union) score.  

