#include <MeanShift.h>
#include <opencv2/core.hpp>

using namespace cv;

class HandsSegmentation {
  private:
    Mat image;
    vector<Rect> handRois;
    int ms_spatial_bandwidth;
    double ms_color_bandwidth;

    vector<Cluster> clusters;

    Mat MSSegment(const Mat &input, const int &spatial_bandwidth, const double &color_bandwidth);
    Mat GetSkinMask();
    Cluster SelectLargestCluster(const vector<Cluster> clusters);
    Mat CreateMaskFromCluster(const Cluster &max_cluster, const Size &image_size);
    vector<Sample> GetSamples(const Mat &from);

    void ShowClusterizedImage(const Mat &image, const vector<Cluster> &clusters);

  public:
    HandsSegmentation(const Mat &_input, const vector<Rect> _rois, const int &_spatial_bandwidth, const double &_color_bandwidth) {
        image = _input;
        handRois = _rois;
        ms_spatial_bandwidth = _spatial_bandwidth;
        ms_color_bandwidth = _color_bandwidth;
    }

    Mat DrawSegments();

    Mat mask;
};