#include <opencv2/core.hpp>

using namespace cv;

class Segmentation {
  public:
    static Mat ClusterWithMeanShift(const Mat &input);
    static Mat SegmentByColor(const Mat &input);
};