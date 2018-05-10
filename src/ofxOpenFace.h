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

// A class for sharing tracking data for a single face
class OpenFaceDataSingleFace {
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

// A class for sharing tracking data for multiple faces
class OpenFaceDataMultipleFaces {
    public:
        vector<OpenFaceDataSingleFace>  vFaces;
};

class ofxOpenFace : public ofThread {
    public:
        ofxOpenFace();
        ~ofxOpenFace();
        void setup(bool bTrackMultipleFaces, int nWidth, int nHeight, bool bUseHOGSVM);
        void setImage(ofImage img);
        void drawFaceIntoMaterial(cv::Mat& mat, const OpenFaceDataSingleFace& data);
        void exit();
        void stop();
        void resetFaceModel();
        int getFPS();
    
        static ofEvent<OpenFaceDataSingleFace>            eventDataReadySingleFace;
        static ofEvent<OpenFaceDataMultipleFaces>         eventDataReadyMultipleFaces;
    
    private:
        void setupSingleFace();
        void setupMultipleFaces(bool bUseHOGSVM);
        void processImageSingleFace();
        void processImageMultipleFaces();
        virtual void threadedFunction();
        void setFPS(float value);
        void drawGazes(cv::Mat& mat, const OpenFaceDataSingleFace& data);
    
        static void NonOverlapingDetections(const vector<LandmarkDetector::CLNF>& clnf_models, vector<cv::Rect_<double>>& face_detections);
    
        int                                             fx, fy, cx, cy;
        int                                             nImgWidth;   // the width of the image used for tracking
        int                                             nImgHeight;  // the height of the image used for tracking
        int                                             nMaxFaces; // the maximum number of faces
        int                                             nFrameCount; // count the frames being tracked
        vector<LandmarkDetector::CLNF>                  vFace_models;
        LandmarkDetector::CLNF                          face_model;
        vector<bool>                                    vActiveModels;
        LandmarkDetector::FaceModelParameters           det_parameters;
        vector<LandmarkDetector::FaceModelParameters>   vDet_parameters;
        bool                                            bExit = false; // flag to close the thread
        float                                           fFPS = 0.0f; // thread frame rate
        ofMutex                                         mutexFPS;
        ofMutex                                         mutexImage;
        float                                           fTimePerRunMs = 0.0f;
        bool                                            bMultipleFaces;
        bool                                            bHaveNewImage = false; // there is a new image available
        cv::Mat                                         matToProcessColor; // the material to process for tracking
        cv::Mat                                         matToProcessGrayScale; // the material to process for tracking
    
    
        const int                                       draw_shiftbits = 4;
        const int                                       draw_multiplier = 1 << 4;
};
