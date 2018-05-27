#include "ofxCv.h"

#pragma once

// A class for storing the raw data from OpenFace for a single face
class ofxOpenFaceDataSingleFace {
public:
    bool                    detected = false;
    cv::Point3f             gazeLeftEye;
    cv::Point3f             gazeRightEye;
    cv::Vec6d               pose;
    vector<cv::Point2f>     allLandmarks2D;
    vector<cv::Point2f>     eyeLandmarks2D;
    vector<cv::Point3f>     eyeLandmarks3D;
    double                  certainty = 0.0f;
    cv::Rect                rBoundingBox;
    string                  sFaceID = "";
    
    //void draw();
};
