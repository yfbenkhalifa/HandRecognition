#include "common.h"
#include "preprocess.h"
#include <filesystem>
#include <iostream>
#include <string>
#include <stdexcept>
#include <stdio.h>
#include "dataset.h"
#include <dirent.h>
#include <iostream>
#include <vector>
#include <fstream>

#include <iostream>

#include <Python.h>
using namespace std;
using namespace cv;

string pytohnCommand = "python3 ../python/main.py ";
string activeDir = "../activeDir/";
string saveDir = "../results/handDetection/";
int detectionWidth, detectionHeight;
int detectOffsetX, detectOffsetY;
int maxHands = 4;
double scoreThreshold = 0.95;

vector<Scalar> colors = {(255,255,0), (255,0,0), (0,0,255), (0,255,0), (255,0,255), (0,255,255), (150,200,50)};

string exec(string command) {

   char buffer[128];
   string result = "";

   // Open pipe to file
   FILE* pipe = popen(command.c_str(), "r");
   if (!pipe) {
      return "popen failed!";
   }

   // read till end of process:
   while (!feof(pipe)) {

      // use buffer to read and add to result
      if (fgets(buffer, 128, pipe) != NULL)
         result += buffer;
   }

   pclose(pipe);
   return result;
}



vector<HandMetadata> detect(Mat src) {
    //Takes as argument the image path and reads it 
    //Mat src = imread(srcPath);
    //Select detection parameters
    imwrite(activeDir + "src.jpg", src);

    string command = pytohnCommand;
    string result = exec(command);
    cout << result << endl;
    vector<int> handROICoords = readCSV(activeDir + "test.csv");
    //Perform detection
    vector<HandMetadata> hands;
    for(int i=0; i<handROICoords.size(); i+=4){
        vector<int> vec {&handROICoords[i], &handROICoords[i+4]};
        HandMetadata temp = loadHandMetadata(vec);
        hands.push_back(temp);
    }
    
    //cout << "Found " + to_string(hands.size()) + " hand" << endl;
    
    
    return hands;
}

Mat drawROI(Mat src, vector<HandMetadata> hands){
    Mat dst = src.clone();
    int colorId = 0;
    int thickness = 2;
    for(auto hand : hands){
        Point p1(hand.PosX, hand.PosY);
        Point p2(hand.PosX + hand.Width, hand.PosY + hand.Height);
        rectangle(dst, p1, p2,
                colors.at(colorId),
                thickness, LINE_8);
        colorId += 1;
        if(colorId == colors.size()) colorId = 0;
    }
    
    return dst;
}

Mat handDetectionModule(Mat src){
    //PreprocessImage(src);
    
    vector<HandMetadata> hands = detect(src);
    Mat dst = drawROI(src, hands);
    // imshow("result", dst);
    // waitKey(0);
    return dst;
}


int main(int argcc, char **argv){
    //vector<Mat> src = readSrcImages("../dataset/benchmark/rgb/", 30);
    int count = 0;
    string image = "10.jpg";
    Mat src = imread("../dataset/benchmark/rgb/" + image, IMREAD_ANYCOLOR);
    cout << "Processing Image" << endl;
    Mat dst = handDetectionModule(src);
    cout << "Saving Image" << endl;
    imwrite(saveDir + image, dst);
    std::remove("../activeDir/test.csv");
    std::remove("../activeDir/test.jpg");
    std::remove("../activeDir/src.jpg");
    
    //createDataset(500);
    return -1;
}

