// ofxOpenFace0.h
//
// based on:
//   https://github.com/TadasBaltrusaitis/OpenFace/blob/master/exe/FaceLandmarkVid/FaceLandmarkVid.cpp
//   https://github.com/TadasBaltrusaitis/OpenFace/blob/master/exe/FaceLandmarkVidMulti/FaceLandmarkVidMulti.cpp
//

#pragma once

#include <ImageManipulationHelpers.h>
#include "LandmarkCoreIncludes.h"
#include "GazeEstimation.h"

#include <Visualizer.h>
#include <VisualizationUtils.h>

#include "ofxCv.h"
#define OPENFACE_N_KEYPOINTS 68

#ifndef OPENFACE_USE_MULTI
#define OPENFACE_USE_MULTI false
#endif

#ifndef OPENFACE_WIDTH
#define OPENFACE_WIDTH 640
#endif

#ifndef OPENFACE_HEIGHT
#define OPENFACE_HEIGHT 480
#endif

using namespace std;


namespace ofxOpenFace{

    // data structure for a detected face
    class faceData{public:
        cv::Vec6d pose;
        cv::Mat_<float>* landmarks;
        cv::Point3f gaze0;
        cv::Point3f gaze1;
        float certainty;
        bool active = false;
        cv::Rect_<float> rect;
        
        // get facial landmark coordinate by index (0-67)
        ofVec2f getLandmark(int i){
            if (active == false){
                return ofVec2f(-1,-1); //err no face
            }
            int n = landmarks->rows/2;
            if (n != OPENFACE_N_KEYPOINTS){
                return ofVec2f(-1,-1); //err no face
            }
            float x = landmarks->at<float>(i);
            float y = landmarks->at<float>(i+n);
            return ofVec2f(x,y);
        }
        
        // draw debug info
        void draw(ofColor col = ofColor::yellow){
            if (active){
                ofNoFill();
                ofSetColor(col); ofSetLineWidth(5);
                ofDrawRectangle(rect.x, rect.y, rect.width, rect.height);
                ofFill();
                for (int i = 0; i < OPENFACE_N_KEYPOINTS; i++){
                    ofVec2f pt = getLandmark(i);
                    ofSetColor(0);  ofDrawCircle(pt,4);
                    ofSetColor(col);ofDrawCircle(pt,2);
                }
            }
        }
    };


    // OpenFace uses a sequence reader class to read frames from video
    // here we fool it with a fake one since OF is handling the input
    class fakeReader{public:
        int fx = OPENFACE_WIDTH;
        int fy = OPENFACE_HEIGHT;
        int cx = OPENFACE_WIDTH/2;
        int cy = OPENFACE_HEIGHT/2;
        cv::Mat_<uchar> GetGrayFrame(cv::Mat latest_frame){
            cv::Mat_<uchar> latest_gray_frame = cv::Mat();
            Utilities::ConvertToGrayscale_8bit(latest_frame, latest_gray_frame);
            return latest_gray_frame;
        }
    };

    // Base class
    class BaseFace{
      protected:
        cv::Mat mat_rgb;
        cv::Mat mat_bgr;
        ofPixels ofpix;

      public:
        cv::Mat canvas;
        float fps;
 
        string model_location = "";
        
        fakeReader sequence_reader;

        LandmarkDetector::CLNF face_model;
        LandmarkDetector::FaceModelParameters det_parameters;
        Utilities::FpsTracker fps_tracker;
        Utilities::Visualizer visualizer = Utilities::Visualizer(true, false, false, false);
        
        vector<string> get_arguments(int argc, char **argv){
            vector<string> arguments;
            for (int i = 0; i < argc; ++i){
                arguments.push_back(string(argv[i]));
            }
            return arguments;
        }
        
        void setup(){
            vector<string> arguments; arguments.push_back("");
            det_parameters = LandmarkDetector::FaceModelParameters(arguments);
            
            if (model_location == ""){
                char cwd_buffer[1024];
                char *cwd = getcwd(cwd_buffer, sizeof(cwd_buffer));
                model_location = ofToString(cwd)+"/../../../data";
            }
            det_parameters.model_location = model_location+"/model/main_ceclm_general.txt";
            cout << "ASSUMED MODEL_LOCATION: " << model_location << endl;
        }
    };
}
// SINGLE FACE -------------------------------------------------------------
#include "ofxOpenFaceSingle.h"

// MULTI FACE -------------------------------------------------------------
#include "ofxOpenFaceMulti.h"

// TRACKING -------------------------------------------------------------
#include "ofxOpenFaceTracker.h"

// THREADING -------------------------------------------------------------
#include "ofxOpenFaceThread.h"
