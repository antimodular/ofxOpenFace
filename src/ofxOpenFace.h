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
#include <SequenceCapture.h>

// Libraries for landmark detection (includes CLNF and CLM modules)
#include "LandmarkCoreIncludes.h"
#include "GazeEstimation.h"

#include <fstream>
#include <sstream>

// OpenCV includes
#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Visualizer.h>
#include <VisualizationUtils.h>

#pragma once

class OpenFaceData {
    public:
        bool                    detected = false;
        cv::Point3f             gazeLeftEye;
        cv::Point3f             gazeRightEye;
        cv::Vec6d               pose;
        vector<cv::Point2d>     eyeLandmarks2D;
        vector<cv::Point3d>     eyeLandmarks3D;
        double                  certainty = 0.0f;
};

class ofxOpenFace : public ofThread {
    public:
        ofxOpenFace();
        ~ofxOpenFace();
        void setup(int nWidth, int nHeight);
        void processImage(cv::Mat mat);
        void setImage(ofImage img);
        void setImage(cv::Mat mat);
        void setImage(ofPixels pix);
        void draw(int x, int y);
        void exit();
        void stop();
        void resetFaceModel();
        const OpenFaceData& getFaceData();
        int getFPS();
    
    private:
        virtual void                            threadedFunction();
        void                                    setFPS(float value);
    
        int                                     nImgWidth;   // the width of the image used for tracking
        int                                     nImgHeight;  // the height of the image used for tracking
        ofImage                                 imgGrayScale;
        ofImage                                 imgVisualized;
        LandmarkDetector::CLNF*                 pFace_model = nullptr;
        LandmarkDetector::FaceModelParameters*  pDet_parameters = nullptr;
        Utilities::Visualizer*                  pVisualizer = nullptr;
        OpenFaceData                            faceData;
        bool                                    bExit = false; // flag to close the thread
        float                                   fFPS = 0.0f; // thread frame rate
        ofMutex                                 mutexFPS;
        ofMutex                                 mutexImage;
        ofMutex                                 mutexVisualizer;
        float                                   fTimePerRunMs = 0.0f;
        bool                                    bHaveNewImage = false; // there is a new image available
        cv::Mat                                 matToProcess; // the material to process for tracking
};
