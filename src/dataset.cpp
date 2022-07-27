#include "dataset.h"
#include "preprocess.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <map>
#include <iostream>
#include <vector>
#include <string>
#include <iostream>



Mat sharpen(Mat src){
    Mat dst = src.clone();
    Laplacian(src, dst, CV_16S);
    return dst;
}


Image::Image(Mat src, Mat mask, vector<int> coords){
    this->src = src;
    this->mask = mask;
    this->coords = coords;
}

HandMetadata::HandMetadata(){
}

void Image::preProcessImages(){
    Mat src_gray;
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    GaussianBlur( src, src, Size(3, 3), 0, 0, BORDER_DEFAULT );
    cvtColor( src, src_gray, COLOR_BGR2GRAY ); 
    Mat abs_dst, dst;
    Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
    
    convertScaleAbs( src, abs_dst );
}

void Image::applyCanny(){
    Canny(src, cannySrc, 20, 20);
}

void Image::segmentImage(){
    Vec3b threshold = (200,200,200);
    Vec3b nullValue = (0,0,0);
    for(int h = 0; h < handSrc.size(); h++){
        Mat dst = handSrc.at(h).clone();
        for(int i=0; i<handSrc.at(h).rows; i++){
            for(int j=0; j<handSrc.at(h).cols; j++){
                //cout << mask.at<Vec3b>(i,j) << "\n";
                if(!inRange(handMasks.at(h).at<Vec3b>(i,j), threshold))
                    dst.at<Vec3b>(i,j) = nullValue;
            }
        }
        handFinals.push_back(dst);
    }
    
}

bool ishand(HandMetadata hand, int x,int y,int w, int h){
    Rect handBox (hand.PosX, hand.PosY, hand.Width, hand.Height);
    Rect srcBox (x,y,w,h);
    return ((handBox & srcBox).area() > 0);
}

void Image::cutImage(){
    for(auto hand : handsMetadata){
        if(!hand.isNull && !hand.isInvalid){
            cv::Range rows(hand.PosY,hand.PosY + hand.Height);
            cv::Range cols(hand.PosX, hand.PosX + hand.Width);
    
            handSrc.push_back(src(rows, cols));
            handCanny.push_back(cannySrc(rows, cols));
            handMasks.push_back(mask(rows,cols));
        }
    }
}


void Image::cutBackground(){
    int width = 250;
    int height = 200;

    for(int i = 0; i<src.rows-width; i+=width){
        for(int j = 0; j < src.cols-height; j+=height){
            bool checkhand = false;
            for(auto hand : handsMetadata){
                if(ishand(hand, j,i,width, height))
                    checkhand = true;
            }
            if(!checkhand){
                cv::Range rows(i,i + height);
                cv::Range cols(j,j+width);
                nothands.push_back(src(rows,cols));
                nothandsCanny.push_back(cannySrc(rows, cols));
            }
            
        }
    }
}

void Image::loadCoordinates(){
    //cout << coords.size() << endl;
    for(int i=0; i<coords.size(); i+=4){
        HandMetadata temp;
        temp.PosX = coords.at(i);
        temp.PosY = coords.at(i+1);
        temp.Width = coords.at(i+2);
        temp.Height = coords.at(i+3);
        if(temp.PosY == 0 && temp.PosX == 0 && temp.Width == 0 && temp.Height == 0 ) temp.isNull = true;
        else temp.isNull = false;
        if(temp.PosY + temp.Height > src.rows || temp.PosX + temp.Width > src.cols) temp.isInvalid = true;
        else temp.isInvalid = false;
        if(!temp.isNull && !temp.isInvalid){
            handsMetadata.push_back(temp);
        }
        
    } 
    //cout << handsMetadata.size() << endl;
    
}

bool computeMask(Mat src, Mat dst, Mat mask){
    Vec3b threshold = (200,200,200);
    Vec3b nullValue = (0,0,0);
    for(int i=0; i<src.cols; i++){
        for(int j=0; j<src.rows; j++){
            if(!inRange(mask.at<Vec3b>(i,j), threshold))
                dst.at<Vec3b>(i,j) = nullValue;
        }
    }
    return true; 
}

bool inRange(Vec3b vec, Vec3b threshold){
    for(int i=0; i<vec.channels; i++){
        if(vec[i] < threshold[i])
            return false;
    }
    return true;
}

Mat applyMask(Mat src, Mat mask){
    Mat dst = src.clone();
    Vec3b threshold = (200,200,200);
    Vec3b nullValue = (0,0,0);
    for(int i=0; i<src.rows; i++){
        for(int j=0; j<src.cols; j++){
            //cout << mask.at<Vec3b>(i,j) << "\n";
            if(!inRange(mask.at<Vec3b>(i,j), threshold))
                dst.at<Vec3b>(i,j) = nullValue;
        }
    }
    return dst;
    
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

string charToString(char *c){
    string s;
    while(*c != '\0'){
        s += *c;
        c++;
    }
    return s;
}

vector<string> readPath(char *folderPath){
    DIR *dir; struct dirent *diread;
    vector<string> files;

    if ((dir = opendir(folderPath)) != nullptr) {
        while ((diread = readdir(dir)) != nullptr) {
            string folderName = diread->d_name;
            if(folderName != "." && folderName != ".." && folderName.find("Identifier") == std::string::npos){
                //cout << "opened " + folderName << endl; 
                files.push_back(folderName);
            }
                
        }
        closedir (dir);
    } else {
        cout << folderPath << " ";
        perror ("opendir");
    }

    sort(files.begin(), files.end());

    return files;
}

vector<Mat> readSrcImages(char *folderPath, int maxSize){
    vector<string> files  = readPath(folderPath);
    vector<Mat> src;
    Mat temp;
    int count = 0;
    for (auto file : files) {
        string filePath = charToString(folderPath) + '/' + file;
        if (filePath.find(".jpg") != std::string::npos) {
            if(filePath.find("mask") == std::string::npos){
                src.push_back(imread(filePath, IMREAD_COLOR));
                vector<string> str = splitstr(filePath, ".");
                //cout << "Added " + file << endl;
            }else{
                cout << "Invalid image format";
            }
        }
        count ++;
        if(count >= maxSize) break;
    }
    return src;
} 

vector<Mat> readMaskImages(char *folderPath, int maxSize){
    vector<string> files  = readPath(folderPath);
    vector<Mat> masks;
    Mat temp;
    int count = 0;
    for (auto file : files) {
        string filePath = charToString(folderPath) + '/' + file;
        if (filePath.find(".jpg") != std::string::npos) {
            if(filePath.find("mask") != std::string::npos){
                masks.push_back(imread(filePath, IMREAD_COLOR));
                //cout << "Added mask" + file << endl;
            }else{
                cout << "Invalid image format"; 
            }
        }
        count++;
        if (count >= maxSize) break;
    }
    return masks;
} 

vector<vector<int>> readCsvFiles(char *folderPath, int maxSize){
    vector<string> files  = readPath(folderPath);
    vector<vector<int>> csv;
    vector<int> temp;
    int count = 0;
    for (auto file : files) {
        string filePath = charToString(folderPath) + '/' + file;
        if (filePath.find(".csv") != std::string::npos) {
            csv.push_back(readCSV(filePath));
        }else{cout << "Invalid file formar" << endl;}
        count++;
        if(count >= maxSize) break;
    }
    
    return csv;
}


string vectorToString(vector<int> vec){
    string s;
    for(auto x : vec)
        s += x;
    return s;
}

HandMetadata loadHandMetadata(vector<int> coords){
    HandMetadata temp;
    temp.PosX = coords.at(0);
    temp.PosY = coords.at(1);
    temp.Width = coords.at(2);
    temp.Height = coords.at(3);
    return temp;
}


//Assume the bounding box is rectangular
Mat cutImage(Mat src, HandMetadata boundingBox){
    cv::Range rows(boundingBox.PosY,boundingBox.PosY + boundingBox.Height);
    cv::Range cols(boundingBox.PosX, boundingBox.PosX + boundingBox.Width);
    // printf("%i - %i", src.rows, boundingBox.PosY + boundingBox.Height);
    // printf("\n%i - %i", src.cols, boundingBox.PosX + boundingBox.Width);
    Mat rez = src(rows, cols); 
    return rez;
}

vector<string> splitstr(string str, string deli)
{
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

string readFileIntoString(const string& path) {
    auto ss = ostringstream{};
    ifstream input_file(path);
    if (!input_file.is_open()) {
        cerr << "Could not open the file - '"
             << path << "'" << endl;
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
        //cout << record << endl;
        while (std::getline(line, record, delimiter)) {
            items.push_back(stoi(record));
        }
    }

    return items;
}


void createDataset(int maxSize){
    vector<Mat> images;
    char *pathSrc = "../dataset/training/src";
    char *pathMask = "../dataset/training/mask";
    char *pathCsv = "../dataset/training/csv";
    string destHandpath = "../dataset/processed/hand/";
    string destNotHandpath = "../dataset/processed/notHand/";
    string testHandPath = "../dataset/test/hand/";
    string testNotHandPath = "../dataset/test/notHand/";
    string maskedHandPath = "../dataset/processed/masked/hand";
    string cannyHandDest = "../dataset/canny/training/hand/";
    string cannyNothandDest = "../dataset/canny/training/notHand/";
    string testcannyHandDest = "../dataset/canny/test/hand/";
    string testcannyNothandDest = "../dataset/canny/test/notHand/";
    vector<Mat> srcImages;
    vector<Mat> masks;
    vector<vector<int>> coords;
    double trainingDatasetSize = 0.8;
    double testDatasetSize = 0.2;

    //Creating training dataset
    vector<Image> trainingDataset;

    srcImages = readSrcImages(pathSrc, maxSize);
    masks = readMaskImages(pathMask, maxSize);
    coords = readCsvFiles(pathCsv, maxSize);

    int datasetSize = srcImages.size(); 
    int i, j;
    for(i = 0; i<datasetSize*trainingDatasetSize; i++){
        Image temp(srcImages.at(i), masks.at(i), coords.at(i));
        temp.loadCoordinates();
        temp.preProcessImages();
        temp.applyCanny();
        temp.cutImage();
        temp.segmentImage();
        temp.cutBackground();
        trainingDataset.push_back(temp);
    }
    cout << i << endl;

    //Creating test dataset
    vector<Image> testDataset;

    cout << datasetSize << endl;
    for(j = i; j<datasetSize; j++){
        Image temp(srcImages.at(j), masks.at(j), coords.at(j));
        temp.loadCoordinates();
        temp.preProcessImages();
        temp.applyCanny();
        temp.cutImage();
        temp.segmentImage();
        temp.cutBackground();
        testDataset.push_back(temp);
    }
    cout << testDataset.size() << endl;

    //Preprocess 
    for(int l = 0; l < trainingDataset.size(); l++){
        for(int h = 0; h < trainingDataset.at(l).handFinals.size(); h++) 
            imwrite(maskedHandPath + to_string(l) + to_string(h) + ".jpg", trainingDataset.at(l).handFinals.at(h));
        for(int h = 0; h < trainingDataset.at(l).handSrc.size(); h++) 
            imwrite(destHandpath + to_string(l) + to_string(h) + ".jpg", trainingDataset.at(l).handSrc.at(h));
        for(int h = 0; h < trainingDataset.at(l).handSrc.size(); h++) 
            imwrite(cannyHandDest + to_string(l) + to_string(h) + ".jpg", trainingDataset.at(l).handCanny.at(h));
        for(int h = 0; h < trainingDataset.at(l).handSrc.size(); h++) 
            imwrite(cannyNothandDest + to_string(l) + to_string(h) + ".jpg", trainingDataset.at(l).nothandsCanny.at(h));
        for(int h = 0; h<trainingDataset.at(l).nothands.size(); h++) 
            imwrite(destNotHandpath + to_string(l) + to_string(h) + ".jpg", trainingDataset.at(l).nothands.at(h));
    }
    
    for(int l = 0; l < testDataset.size(); l++){
        for(int h = 0; h < testDataset.at(l).handFinals.size(); h++) 
            imwrite(testHandPath + to_string(l) + to_string(h) + ".jpg", testDataset.at(l).handFinals.at(h));
        for(int h = 0; h < testDataset.at(l).handSrc.size(); h++) 
            imwrite(testHandPath + to_string(l) + to_string(h) + ".jpg", testDataset.at(l).handSrc.at(h));
        for(int h = 0; h<testDataset.at(l).nothands.size(); h++) 
            imwrite(testNotHandPath + to_string(l) + to_string(h) + ".jpg", testDataset.at(l).nothands.at(h));
        for(int h = 0; h < testDataset.at(l).handSrc.size(); h++) 
            imwrite(testcannyHandDest + to_string(l) + to_string(h) + ".jpg", testDataset.at(l).handCanny.at(h));
        for(int h = 0; h < trainingDataset.at(l).handSrc.size(); h++) 
            imwrite(testcannyNothandDest + to_string(l) + to_string(h) + ".jpg", trainingDataset.at(l).nothandsCanny.at(h));
    }
}