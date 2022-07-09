#include "common.h"



Mat segment(Mat, Vec3b, Vec3b);

/* Image Pre-processing Functions */

// Returns smoothed image using Gaussian bluring with given Kernel and value for sigma
Mat smoothImage (Mat, Mat, double);

// Returns sharpened image using derivative fitler with given kernel
Mat sharpenImage (Mat, Mat, double);


/* Utility Functions */ 

// Returns true if the pixel BGR intensity is within the given thresholds 
bool isWithin(Vec3b, Vec3b, Vec3b);

class Preprocess
{
private:
    int private_variable;

public:
    int getPrivate();
};