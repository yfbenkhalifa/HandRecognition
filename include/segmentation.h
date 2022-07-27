#include <opencv2/core.hpp>

using namespace cv;

class Segmentation {
  public:
    static Mat ClusterWithMeanShift(const Mat &input, const int &spatial_bandwidth, const int &color_bandwidth);
    static Mat SegmentByColor(const Mat &input);
    static Mat GetSkinMask(const Mat &input);
};