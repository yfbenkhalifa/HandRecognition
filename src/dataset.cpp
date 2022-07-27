#include "dataset.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

void Image::segmentImage() {
    Vec3b threshold = (200, 200, 200);
    Vec3b nullValue = (0, 0, 0);
    for (int h = 0; h < handSrc.size(); h++) {
        Mat dst = handSrc.at(h).clone();
        for (int i = 0; i < handSrc.at(h).rows; i++) {
            for (int j = 0; j < handSrc.at(h).cols; j++) {
                // cout << mask.at<Vec3b>(i,j) << "\n";
                if (!inRange(handMasks.at(h).at<Vec3b>(i, j), threshold))
                    dst.at<Vec3b>(i, j) = nullValue;
            }
        }
        handFinals.push_back(dst);
    }
}

void Image::cutImage() {
    for (auto hand : handsMetadata) {
        if (!hand.isNull && !hand.isInvalid) {
            cv::Range rows(hand.PosY, hand.PosY + hand.Height);
            cv::Range cols(hand.PosX, hand.PosX + hand.Width);
            // cout << hand.PosY << endl;
            // cout << hand.PosX << endl;
            // Mat tempSrc = src.clone();
            // Mat tempMask = mask.clone();
            handSrc.push_back(src(rows, cols));
            handMasks.push_back(mask(rows, cols));
        }
    }
}

void Image::loadCoordinates() {
    for (int i = 0; i < coords.size(); i += 4) {
        HandMetadata temp;
        temp.PosX = coords.at(i);
        temp.PosY = coords.at(i + 1);
        temp.Width = coords.at(i + 2);
        temp.Height = coords.at(i + 3);
        if (temp.PosY == 0 && temp.PosX == 0 && temp.Width == 0 && temp.Height == 0)
            temp.isNull = true;
        else
            temp.isNull = false;
        if (temp.PosY + temp.Height > src.rows || temp.PosX + temp.Width > src.cols)
            temp.isInvalid = true;
        else
            temp.isInvalid = false;
        handsMetadata.push_back(temp);
    }
    cout << handsMetadata.size() << endl;
}

bool computeMask(Mat src, Mat dst, Mat mask) {
    Vec3b threshold = (200, 200, 200);
    Vec3b nullValue = (0, 0, 0);
    for (int i = 0; i < src.cols; i++) {
        for (int j = 0; j < src.rows; j++) {
            if (!inRange(mask.at<Vec3b>(i, j), threshold))
                dst.at<Vec3b>(i, j) = nullValue;
        }
    }
    return true;
}

bool inRange(Vec3b vec, Vec3b threshold) {
    for (int i = 0; i < vec.channels; i++) {
        if (vec[i] < threshold[i])
            return false;
    }
    return true;
}

Mat applyMask(Mat src, Mat mask) {
    Mat dst = src.clone();
    Vec3b threshold = (200, 200, 200);
    Vec3b nullValue = (0, 0, 0);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            // cout << mask.at<Vec3b>(i,j) << "\n";
            if (!inRange(mask.at<Vec3b>(i, j), threshold))
                dst.at<Vec3b>(i, j) = nullValue;
        }
    }
    return dst;
}

void subMatrix(Mat src, Mat dst, int posX, int posY, int sizeX, int sizeY) { dst = src(Rect(posX, posY, sizeX, sizeY)); }

string charToString(char *c) {
    string s;
    while (*c != '\0') {
        s += *c;
        c++;
    }
    return s;
}

Image readImageFiles(char *folderPath, int id) {
    DIR *dir;
    struct dirent *diread;
    vector<string> files;

    if ((dir = opendir(folderPath)) != nullptr) {
        while ((diread = readdir(dir)) != nullptr) {
            string folderName = diread->d_name;
            if (folderName != "." && folderName != "..") {
                cout << "opened " + folderName << endl;
                files.push_back(folderName);
            }
        }
        closedir(dir);
    } else {
        cout << folderPath << " ";
        perror("opendir");
    }

    Image temp;
    for (auto file : files) {
        string filePath = charToString(folderPath) + '/' + file;
        int count = 1;
        if (filePath.find(".jpg") != std::string::npos) {
            if (filePath.find("mask") != std::string::npos) {
                temp.mask = imread(filePath, IMREAD_GRAYSCALE);
                // cout << "Added mask" + file << endl;
            } else {
                temp.src = imread(filePath, IMREAD_COLOR);
                // cout << "Added src" + file << endl;
            }
        } else if (filePath.find(".csv") != std::string::npos) {
            temp.coords = readCSV(filePath);
            // cout << "reading csv" << endl;
        }
        // cout << "Added " + filePath << endl;
        // cout << charToString(folderPath) + '/' + file << endl;
    }

    return temp;
}

string vectorToString(vector<int> vec) {
    string s;
    for (auto x : vec)
        s += x;
    return s;
}

HandMetadata loadHandMetadata(vector<int> coords) {
    HandMetadata temp;
    temp.PosX = coords.at(0);
    temp.PosY = coords.at(1);
    temp.Width = coords.at(2);
    temp.Height = coords.at(3);
    return temp;
}

// Assume the bounding box is rectangular
Mat cutImage(Mat src, HandMetadata boundingBox) {
    cv::Range rows(boundingBox.PosY, boundingBox.PosY + boundingBox.Height);
    cv::Range cols(boundingBox.PosX, boundingBox.PosX + boundingBox.Width);
    // printf("%i - %i", src.rows, boundingBox.PosY + boundingBox.Height);
    // printf("\n%i - %i", src.cols, boundingBox.PosX + boundingBox.Width);
    Mat rez = src(rows, cols);
    return rez;
}

vector<string> splitstr(string str, string deli) {
    vector<string> vec;
    int start = 0;
    int end = str.find(deli);
    while (end != -1) {
        vec.push_back(str.substr(start, end - start));
        start = end + deli.size();
        end = str.find(deli, start);
    }
    return vec;
}

string readFileIntoString(const string &path) {
    auto ss = ostringstream{};
    ifstream input_file(path);
    if (!input_file.is_open()) {
        cerr << "Could not open the file - '" << path << "'" << endl;
        exit(EXIT_FAILURE);
    }
    ss << input_file.rdbuf();
    return ss.str();
}

std::vector<int> readCSV(string path) {
    string filename(path);
    string file_contents;
    char delimiter = ',';

    file_contents = readFileIntoString(filename);

    istringstream sstream(file_contents);
    std::vector<int> items;
    string record;

    int counter = 0;
    while (std::getline(sstream, record)) {
        istringstream line(record);
        while (std::getline(line, record, delimiter)) {
            items.push_back(stoi(record));
        }
    }

    return items;
}
