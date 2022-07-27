#include <opencv2/core.hpp>

using namespace cv;

class Segmentation
{
public:
    static Mat ClusterWithMeanShift(Mat input);
};