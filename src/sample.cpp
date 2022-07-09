#include "sample.h"
#include <math.h>

double euclidean_distance_square(const vector<double> &point_a, const vector<double> &point_b) {
    double total = 0;
    for (int i = 0; i < point_a.size(); i++) {
        const double temp = (point_a[i] - point_b[i]);
        total += temp * temp;
    }
    return (total);
}

double euclidean_distance(const vector<double> &point_a, const vector<double> &point_b) { return sqrt(euclidean_distance_square(point_a, point_b)); }

double Sample::distanceFrom(const Sample &other) const { return sqrt(euclidean_distance_square(color, other.color) + euclidean_distance_square(location, other.location)); }
