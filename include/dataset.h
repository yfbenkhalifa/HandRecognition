#include "common.h"
#include "preprocess.h"
#include <dirent.h>


using std::cin;
using std::cout;
using std::endl;
using std::vector;

using namespace std;

class HandMetadata {
  public:
    int PosX;
    int PosY;
    int Width;
    int Height;
    Rect BoundingBox;
    bool isNull;
    bool isInvalid;

    HandMetadata();
    void initBoundingBox();

    Rect getBoundingBox() { return Rect(PosX, PosY, Width, Height); }
};

class Image {
  public:
    int id;
    string jpgFileName;
    string maskFileName;
    string csvFileName;
    Mat src;
    Mat mask;
    Mat cannySrc;
    vector<int> coords;
    vector<HandMetadata> handsMetadata;
    vector<Mat> handSrc;
    vector<Mat> handMasks;
    vector<Mat> handFinals;
    vector<Mat> nothands;
    vector<Mat> nothandsCanny;
    vector<Mat> handCanny;

    Image(Mat, Mat, vector<int>);
    void loadCoordinates();
    void cutImage();
    void applyCanny();
    void segmentImage();
    void preProcessImages();
    void cutBackground();
};

class Dataset {
  public:
    static string vectorToString(vector<int> vec);
    static vector<string> readPath(char *folderPath);
    static bool computeMask(Mat mask);
    // static Image readImageFiles(char* folderPath, int id);
    static vector<Mat> readSrcImages(char *folderPath, int maxSize);
    static vector<Mat> readMaskImages(char *folderPath, int maxSize);
    static vector<vector<int>> readCsvFiles(char *folderPath, int maxSize);
    static void subMatrix(Mat src, Mat dst, int posX, int posY, int sizeX, int sizeY);
    static Mat cutImage(Mat src, HandMetadata boundingBox);
    static vector<string> splitstr(string str, string deli);
    static vector<int> readCSV(string path, char delimiter = ',');
    static bool computeMask(Mat src, Mat dst, Mat mask);
    static string readFileIntoString(const string &path);
    static Mat applyMask(Mat src, Mat mask);
    static void createDataset(int maxSize = 1000);

    static HandMetadata loadHandMetadata(vector<int> coords);

    static vector<int> subVector(vector<int> const &v, int m, int n);
};
