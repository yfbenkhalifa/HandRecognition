#include <algorithm>
#include <vector>

using namespace std;

class Sample {
  private:
  public:
    vector<double> color;
    vector<double> location;

    Sample() {}

    Sample(int color_size, int location_size) {
        color = vector<double>(color_size, 0);
        location = vector<double>(location_size, 0);
    }

    Sample operator*(const double &multiplier) {
        Sample temp(this->color.size(), this->location.size());
        transform(temp.color.begin(), temp.color.end(), temp.color.begin(), [multiplier](double &c) { return c * multiplier; });
        transform(temp.location.begin(), temp.location.end(), temp.location.begin(), [multiplier](double &c) { return c * multiplier; });

        return temp;
    }

    Sample operator/(const double &divider) {
        double multiplier = 1.0 / divider;
        return *this * multiplier;
    }

    Sample operator+(const Sample &other) {
        for (int i = 0; i < this->color.size(); i++)
            this->color[i] += other.color[i];

        for (int i = 0; i < this->location.size(); i++)
            this->location[i] += other.location[i];
    }

    double distanceFrom(const Sample &other) const;
};