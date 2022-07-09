#include "segmentation.h"
#include "MeanShift.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

using namespace cv;
using namespace std;

Mat Segmentation::ClusterWithMeanShift(Mat input) {
    cvtColor(input, input, CV_8UC1);
    resize(input, input, Size(input.cols / 10, input.rows / 10));
    Mat output = Mat(input.size(), CV_8UC1);

    vector<Sample> samples;

    for (int r = 0; r < input.rows; r++)
        for (int c = 0; c < input.cols; c++) {
            byte temp = input.at<byte>(r, c);
            Sample sample;
            sample.color = {(double)r, (double)c, (double)temp};
            sample.location = {(double)c, (double)r};

            samples.push_back(sample);
        }

    MeanShift *c = new MeanShift(samples);
    vector<Cluster> clusters = c->cluster(samples, 2500);

    int n_clusters = (int)clusters.size();

    cout << n_clusters << endl;

    for (int c = 0; c < n_clusters; c++)
        for (int i = 0; i < clusters[c].original_points.size(); i++) {
            vector<double> location = clusters[c].original_points[i].location;
            output.at<byte>(location[1], location[0]) = (byte)(c * 255 / n_clusters);
        }

    imshow("Output", output);
    waitKey(0);

    return output;
}