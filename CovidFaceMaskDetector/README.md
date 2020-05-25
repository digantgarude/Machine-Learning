## Covid Face Mask Detector

[Tutorial From PyImageSearch](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/?__s=rpaxc6xbop4n2z9srxh8)

[Main Original Author's Post on LinkedIn](https://www.linkedin.com/feed/update/urn%3Ali%3Aactivity%3A6655711815361761280/)

- Following is the link to the dataset files that I used.  [Dataset Link](https://github.com/prajnasb/observations/tree/master/experiements/data)

## Usage

- For training the model first download the dataset and then change the path to the dataset in the DATASET_DIR variable in *train_mask_detector.py* file and run.

```bash
python train_mask_detector.py
```

- To directly run the trained model and check the output, just run.

```bash
python mask_detector_frontend.py
```

## File Info

- train_mask_detector.py -> For training the model. (Make sure to change the DATASET_DIR variable).

- mask_detector_frontend.py -> For running the trained model.

- model_info.py -> This file is only to check if the model was trained well. (Created this because sometimes the image data generator gives error in the end and just so that the model is not required to be trained again just to check if it trained well. )

- mask_detector.h5 -> The mask detector model file.


## Steps

- Following are the steps that are followed.

1. Load the dataset
2. Train Mask classifier with tf / keras
3. Save model
4. Load model
5. Detect faces in video
6. Extract each face using open cv dnn 
7. Apply mask classifier to determine if mask is present or not
8. Display results



