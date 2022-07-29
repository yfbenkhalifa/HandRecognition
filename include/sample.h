#pragma once

#include <algorithm>
#include <opencv2/core.hpp>
#include <vector>

using namespace std;
using namespace cv;

class Sample {
  private:
  public:
    vector<double> color;
    vector<double> location;
    vector<double> originalLocation;

    Sample() {}

    Sample(vector<double> _color, vector<double> _location) {
        color = vector<double>(_color.size(), 0);
        location = vector<double>(_location.size(), 0);
        originalLocation = vector<double>(_location.size(), 0);

        for (int i = 0; i < _color.size(); i++)
            color[i] = _color[i];
        for (int i = 0; i < _location.size(); i++)
            location[i] = _location[i];
        for (int i = 0; i < _location.size(); i++)
            originalLocation[i] = _location[i];
    }

    Sample(const Mat &image, int row, int col, int offsetX = 0, int offsetY = 0) {
        Vec3d temp = image.at<Vec3b>(row, col);
        color = {temp[0], temp[1], temp[2]};
        location = {(double)col + offsetX, (double)row + offsetY};
        originalLocation = {(double)col + offsetX, (double)row + offsetY};
    }

    Sample(int color_size, int location_size) {
        color = vector<double>(color_size, 0);
        location = vector<double>(location_size, 0);
        originalLocation = vector<double>(location_size, 0);
    }

    Sample(int color_size, int location_size, vector<double> _originalLocation) {
        color = vector<double>(color_size, 0);
        location = vector<double>(location_size, 0);
        originalLocation = vector<double>(location_size, 0);

        for (int i = 0; i < originalLocation.size(); i++)
            originalLocation[i] = _originalLocation[i];
    }

    Sample(const Sample &_sample) {
        color = vector<double>(_sample.color.size(), 0);
        location = vector<double>(_sample.location.size(), 0);
        originalLocation = vector<double>(_sample.originalLocation.size(), 0);

        for (int i = 0; i < color.size(); i++)
            color[i] = _sample.color[i];

        for (int i = 0; i < location.size(); i++)
            location[i] = _sample.location[i];

        for (int i = 0; i < originalLocation.size(); i++)
            originalLocation[i] = _sample.originalLocation[i];
    }

    Sample operator*(const double &multiplier) {
        Sample temp(*this);
        transform(temp.color.begin(), temp.color.end(), temp.color.begin(), [multiplier](double &c) { return c * multiplier; });
        transform(temp.location.begin(), temp.location.end(), temp.location.begin(), [multiplier](double &c) { return c * multiplier; });

        return temp;
    }

    Sample operator/(const double &divider) {
        double multiplier = 1.0 / divider;
        return *this * multiplier;
    }

    Sample operator+(const Sample &other) {
        Sample temp = Sample(*this);
        for (int i = 0; i < temp.color.size(); i++)
            temp.color[i] += other.color[i];

        for (int i = 0; i < temp.location.size(); i++)
            temp.location[i] += other.location[i];

        return temp;
    }

    Sample operator+=(const Sample &other) {
        for (int i = 0; i < this->color.size(); i++)
            this->color[i] += other.color[i];

        for (int i = 0; i < this->location.size(); i++)
            this->location[i] += other.location[i];

        return *this;
    }

    double distanceFrom(const Sample &other) const;
    double colorDistanceFrom(const Sample &other) const;
    double locationDistanceFrom(const Sample &other) const;
    bool colorInRange(const vector<double> &lower_bound, const vector<double> &upper_bound) const;
};