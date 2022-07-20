#include "preprocess.h"
#include "common.h"

using namespace std;

bool computeMask(Mat src, Mat dst, Mat mask){
    if(src.size() != mask.size())
        return false;
        
    for(int i=0; i<src.cols; i++){
        for(int j=0; j<src.rows; j++){
            if(mask.at<uchar>(i,j) < 200){
                src.at<uchar>(i,j) = 0;
            }
        }
    }
    return true; 
}

void subMatrix(Mat src, Mat dst, int posX, int posY, int sizeX, int sizeY)
{
    dst = Mat(sizeX, sizeY, CV_8U);

    for (int i = 0; i < sizeY; i++)
    {
        for (int j = 0; j < sizeX; j++)
        {
            dst.at<uchar>(i, j) = src.at<uchar>(posX + i, posY + j);
        }
    }
    
}


//Assume the bounding box is rectangular
void cutImage(Mat src, Mat dst, vector<Point2f> vertices){
    int sizeX = vertices.at(1).x - vertices.at(0).x;
    int sizeY = vertices.at(1).y - vertices.at(0).y;

    Mat cut(sizeX, sizeY, CV_8U);
    
    
}