#include "preprocess.h"
#include "common.h"
#include <dirent.h>

using std::cout; using std::cin;
using std::endl; using std::vector;

using namespace std;


class HandMetadata{
    public:
        int PosX;
        int PosY;
        int Width;
        int Height;
        Rect BoundingBox;
        bool isNull;
        bool isInvalid;
        void initBoundingBox();
};

class Image{
    public:
        int id;
        string jpgFileName;
        string maskFileName;
        string csvFileName;
        Mat src;
        Mat mask;
        vector<int> coords;
        vector<HandMetadata> handsMetadata;
        vector<Mat> handSrc;
        vector<Mat> handMasks;
        vector<Mat> handFinals;

        void loadCoordinates();
        void cutImage();
        void segmentImage();

};
string vectorToString(vector<int> vec);
bool computeMask(Mat mask);
Image readImageFiles(char* folderPath, int id);
void subMatrix(Mat src, Mat dst, int posX, int posY, int sizeX, int sizeY);
Mat cutImage(Mat src, HandMetadata boundingBox);
string readFileIntoString(const string& path);
vector<string> splitstr(string str, string deli);
vector<int> readCSV(string path);
string readFileIntoString(const string& path);
Mat applyMask(Mat src, Mat mask);
bool inRange(Vec3b vec, Vec3b threshold);