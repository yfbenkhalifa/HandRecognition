#include "preprocess.h"
#include "common.h"
#include "dataset.h"
#include <iostream>
#include <vector>
#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include<fstream>
using std::vector;
using namespace std;


int main (int argcc, char ** argv){
    vector<Image> dataset;
    vector<Mat> masks;
    

    for(int i=1; i<100; i++){
        char intString[32] ;
        char path [100] = "../dataset/training/";
        sprintf(intString, "%d", i);
        //cout << intString << endl;
        Image temp = readImageFiles(strcat(path, intString), i);
        temp.id = i;
        dataset.push_back(temp);
    }

    for(auto image : dataset){
        image.loadCoordinates();
        image.cutImage();
        image.segmentImage();
        //imshow("hand", image.src);
        //waitKey(0);
        for(auto mask : image.handSrc){
            //cout << image.id << endl;
            //imshow("hand", mask);
            //waitKey(0);
            string saveFileName = "../dataset/processed/" + to_string(image.id) + ".jpg";  
            imwrite(saveFileName, mask);
        }
    }
    // //imshow("test", dataset.at(0).handSrc.size());
    // waitKey(0);
    return -1;
}