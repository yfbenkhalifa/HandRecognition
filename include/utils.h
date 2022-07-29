#pragma once
#include <fstream>

using namespace std;
using namespace cv;

class Utils {
  public:
    static std::vector<int> explode(std::string const &s, char delim) {
        std::vector<int> result;
        std::istringstream iss(s);

        for (std::string token; std::getline(iss, token, delim);) {
            result.push_back(stoi(std::move(token)));
        }

        return result;
    }

    static vector<Rect> getGroundTruthRois(string file) {
        string det_path = "../dataset/det/";
        string det_file = file.substr(15, file.length() - 19);
        det_path.append(det_file).append(".txt");
        ifstream myfile(det_path);

        vector<Rect> rectangles;
        if (myfile.is_open()) {
            string line;
            while (getline(myfile, line)) {
                vector<int> params = explode(line, '\t');
                if (params.size() < 4)
                    params = explode(line, ' ');
                rectangles.push_back(Rect(params[0], params[1], params[2], params[3]));
            }
            myfile.close();
        }

        return rectangles;
    }

    static Mat getGroundTruthMask(string file) {
        string mask_file = file.substr(15, file.length() - 19);
        string mask_path = "../dataset/mask/";
        mask_path.append(mask_file).append(".png");
        return imread(mask_path, IMREAD_GRAYSCALE);
    }

    static void saveOutput(string file, Mat image) {
        string output_file = file.substr(15, file.length() - 15);
        string output_path = "../output/";
        output_path.append(output_file);
        imwrite(output_path, image);
    }

    static Scalar getColor(int index) {
        vector<Scalar> colors = {Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255), Scalar(255, 255, 0), Scalar(255, 0, 255), Scalar(0, 255, 255), Scalar(150, 200, 50)};
        return colors[index];
    }
};
