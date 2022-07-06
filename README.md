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

Image Pre-processing:
- Smoothing -> Noise reduction using Gaussian Filtering;
- Sharpening -> Edge enhance to facilitate the detection;
- Linear Transformations -> Apply rotations, translations and scale changes so that we obtain scale, position and orientation invariant detection;
