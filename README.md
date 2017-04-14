# Data Science Bowl 2017 Competition

Many thanks to all the authors of kernels which enormously helped with data processing, especially at the beginning!
Some functions are based on the kernels by Guido Zuidhof, ArnavJain, Ankasor, Bohdan Pavlyshenko.
my_PreProc function by: https://github.com/orobix/retina-unet.

https://www.kaggle.com/c/data-science-bowl-2017/kernels

#### LUNA:

* Processing:
    * LUNA_2d_merge_preproc.py - processing LUNA files into format for 2D U-Net segmentation model
    * LUNA_3d_merge_preproc.py - processing LUNA files into format for 3D U-Net segmentation model, basically the same as for 2D but resized and padded into 136x168x168 size for every patient (as big as could fit in my 8GB GPU)

* Models:
    * U-Net 2D: training 2D segmentation neural network on LUNA annotated nodules. No need to resize, so no loss of information during this phase,
    but a lot of false positives are generated. Slices are provided to the model independently, so there's loss of information about their relationships.
    On 0.-255. slices and 0./255. masks it got 1.45 validation dice coefficient after around 20h of training on GTX1080.

        * Prepare subsets of 100 patients each for the generator (makes sure that a big enough amount of slices gets shuffled to provide the model with IID data - could be done better, without saving subsets, that's improved in the generator for DSB)
        * Model that was used is specified in the unet_models.py
        * Train using the generator with 2DUNet_train_generator_subsets.py

    * U-Net 3D: training 3D segmentation neural network on LUNA annotated nodules. Here the slices were resized into 136x168x168, so many of the nodules are greatly reduced in size and some even lost but relationship between the slices are retained. Much less false positives.
    On 0.-255. slices and 0./255. masks it got 0.5 validation dice coefficient, training was faster, took around 10h.
    
        * Model trained with 3DUNet_train_generator.py, patients are simply loaded one by one.
        
        
#### DSB:

* Processing:
    * Data is processed in a way analogical to the LUNA dataset in order to make predictions possible.
    * After it's processed, the data is fed into 2D/3D segmentation models trained on LUNA and predicitons are output: 2DUNet_DSB_predict.py for 2D model and 3DUNet_DSB_predict.ipynb for 3D model.
    * Predictions can be compressed using zarr - zarr_compress_2d/3d

* Models:
    * 3D CNN's for classification were created for training on DSB data predicted by 2D/3D U-Net's. Ones for 2D predicitions are called preds2d_* and for 3D predictions preds3d. Expriments with various classifier architectures are in 2 jupyter notebooks.
    Those models did not give good validation results, probably due to resizing of whole lungs, which caused the most important features such as nodules to be lost or significantly reduced.
    * 2D CNN was also created to be trained on 2D U-Net predictions without resizing, so it is fed with slices one by one, where each slice for a patient with cancer is labeled with 1 and without cancer with 0.
    Did not perform well.
    Features were extracted from 2D CNN classifier which did not perform well by itself. An XGBoost model was trained and optimized on extracted features. Validated on patients 1500-1595 it achieved a logloss of 0.55.
