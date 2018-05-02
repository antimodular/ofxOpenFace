/*
* ofxOpenFace.h
* openFrameworks
*
* The interface to the OpenFace library.
*
*/

#include "ofMain.h"
#include "ofxCv.h"

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

class ofxOpenFace {
    public:
        void setup(int nWidth, int nHeight);
        void update(ofImage &img);
        void draw();
        void resetFaceModel();
    
    private:
        int                                     nImgWidth;   // the width of the image used for tracking
        int                                     nImgHeight;  // the height of the image used for tracking
        ofImage                                 imgGrayScale;
        LandmarkDetector::CLNF*                 pFace_model = nullptr;
        LandmarkDetector::FaceModelParameters*  pDet_parameters = nullptr;
        Utilities::Visualizer*                  pVisualizer = nullptr;
};
