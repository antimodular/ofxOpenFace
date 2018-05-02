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
        void setup();
        void update(ofImage &img);
        void draw();
        void resetFaceModel();
    
    private:
        ofImage                                 imgGrayScale;
        LandmarkDetector::CLNF*                 pFace_model = nullptr;
        LandmarkDetector::FaceModelParameters*  pDet_parameters = nullptr;
        Utilities::Visualizer*                  pVisualizer = nullptr;
};
