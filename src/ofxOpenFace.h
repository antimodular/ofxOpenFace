/*
* ofxOpenFace.h
* openFrameworks
*
* The interface to the OpenFace library.
*
*/

#include "ofMain.h"
#include "ofxCv.h"
#include "ofThread.h"

// OpenFace
#include "LandmarkCoreIncludes.h"
#include <VisualizationUtils.h>
#include <Visualizer.h>
#include <SequenceCapture.h>
#include <RecorderOpenFace.h>
#include <RecorderOpenFaceParameters.h>
#include <GazeEstimation.h>
#include <FaceAnalyser.h>

#include <fstream>
#include <sstream>

// OpenCV includes
#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#pragma once

class OpenFaceData {
    public:
        bool                    detected = false;
        cv::Point3f             gazeLeftEye;
        cv::Point3f             gazeRightEye;
        cv::Vec6d               pose;
        vector<cv::Point2d>     allLandmarks2D;
        vector<cv::Point2d>     eyeLandmarks2D;
        vector<cv::Point3d>     eyeLandmarks3D;
        double                  certainty = 0.0f;
};

class ofxOpenFace : public ofThread {
    public:
        ofxOpenFace();
        ~ofxOpenFace();
        void setup(int nWidth, int nHeight);
        void processImage();
        void setImage(ofImage img);
        void exit();
        void stop();
        void resetFaceModel();
        int getFPS();
    
        static ofEvent<OpenFaceData>            eventDataReady;
    
    private:
        virtual void                                    threadedFunction();
        void                                            setFPS(float value);
        static void                                     NonOverlapingDetections(const vector<LandmarkDetector::CLNF>& clnf_models, vector<cv::Rect_<double>>& face_detections);
    
        int                                             fx, fy, cx, cy;
        int                                             nImgWidth;   // the width of the image used for tracking
        int                                             nImgHeight;  // the height of the image used for tracking
        int                                             nMaxFaces; // the maximum number of faces
        int                                             nFrameCount; // count the frames being tracked
        vector<LandmarkDetector::CLNF>                  vFace_models;
        vector<bool>                                    vActiveModels;
        vector<LandmarkDetector::FaceModelParameters>   vDet_parameters;
        Utilities::Visualizer*                          pVisualizer = nullptr;
        FaceAnalysis::FaceAnalyserParameters*           pFace_analysis_params = nullptr;
        FaceAnalysis::FaceAnalyser*                     pFace_analyser = nullptr;
        bool                                            bExit = false; // flag to close the thread
        float                                           fFPS = 0.0f; // thread frame rate
        ofMutex                                         mutexFPS;
        ofMutex                                         mutexImage;
        float                                           fTimePerRunMs = 0.0f;
        bool                                            bHaveNewImage = false; // there is a new image available
        cv::Mat                                         matToProcessColor; // the material to process for tracking
        cv::Mat                                         matToProcessGrayScale; // the material to process for tracking
        bool                                            bDoVisualizer; // set to true to generate the visualizer image
};
