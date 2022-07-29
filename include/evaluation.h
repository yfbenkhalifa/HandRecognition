#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

using namespace cv;

double evaluateMask(const Mat&, const Mat&);

double evaluateBox(vector<int>, vector<int>);
