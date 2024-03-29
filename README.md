# HandRecognition

Benchmark dataset: https://drive.google.com/drive/folders/1ORmMRRxfLHGLKgqHG-1PKx1ZUCIAJYoa?usp=sharing

As a reference, the images in the provided benchmark dataset have been sampled from the following publicly
available datasets:
    • EgoHands: http://vision.soic.indiana.edu/projects/egohands/
    • HandOverFace(HOF): https://drive.google.com/file/d/1hHUvINGICvOGcaDgA5zMbzAIUv7ewDd3

Algorithms we can use:
- CNN;
- SIFT/SURF feature matching;
- Clustering
- Segmentation

Hand-Recognigtion through Clustering:
    Steps to follow:
    - Preprocess image;
    - Compute segmentation using clustering;
    - Detect hands within the clusters;
    - Detect and define Hand cluster centres for left/right    hands;
    - Based on the identified centres, draw the identification box for both left and right hands;
    - Color the hands with different colors for right and left hand. 

Image Pre-processing:
- Smoothing [Gaussian Blur] -> Noise reduction using Gaussian Filtering;
- Sharpening [Derivative Filter] -> Edge enhance to facilitate the detection;
- Linear Transformations -> Apply rotations, translations and scale changes so that we obtain scale, position and orientation invariant detection;


Image Classification: 
- Define two classes: left hand {0}, right rand {1} (assign integers to ease the CNN)
- If not avaiable, preprocess the training set of images in order to extract the dataset of hands, correctly labeld
- Load and classify training images;
- CNN with the following parameters: First try with 1 Convolutional layer, 3 pooling layers and 5 Fully interconnected layers (just to try);


