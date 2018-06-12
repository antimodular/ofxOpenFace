#pragma once

#include "ofMain.h"
#include "ofxOpenFace.h"
#include "ofxXmlSettings.h"
#include "ofxGui.h"

// The application settings
class appSettings {
    public:
        int nCameraIndex;
        int nCameraWidth;
        int nCameraHeight;
        int nCameraFrameRate;
        bool bMultipleFaces;
        bool bDoCvTracking; // true: perform ofxCv tracking of the face for time alive
        int nMaxFaces;
        int nTrackingPersistenceMs; // time allowed for tracking to forget an object
        int nTrackingTolerancePx; // pixels allowed to move for tracking to changes
        float fCertaintyNorm; // normalized certainty below which we do not recognize a face
        int nKillAfterDisappearedMs; // time after which we forget a missing face
        int fx, fy, cx, cy; // camera data
        LandmarkDetector::FaceModelParameters::FaceDetector eDetectorFace; // face detector
        LandmarkDetector::FaceModelParameters::LandmarkDetector eDetectorLandmarks; // landmark detector
};

// The main app
class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
        void draw();
        void exit();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
    
    private:
        void updateGUI();
        void loadSettings();
        void saveSettings();
        void onFaceDataClear(bool& data);
        void onFaceDataSingleRaw(ofxOpenFaceDataSingleFace& data);
        void onFaceDataMultipleRaw(vector<ofxOpenFaceDataSingleFace>& data);
        void onFaceDataSingleTracked(ofxOpenFaceDataSingleFaceTracked& data);
        void onFaceDataMultipleTracked(vector<ofxOpenFaceDataSingleFaceTracked>& data);
    
        ofVideoGrabber                          vidGrabber;
        ofImage                                 imgToProcess;
        ofxOpenFace                             openFace;
        ofxOpenFaceDataSingleFace                  latestDataSingle;
        vector<ofxOpenFaceDataSingleFace>          latestDataMultiple;
        ofxOpenFaceDataSingleFaceTracked           latestDataSingleTracked;
        vector<ofxOpenFaceDataSingleFaceTracked>   latestDataMultipleTracked;
        ofMutex                                 mutexFaceData;
    
        // For the single face GUI
        ofxPanel                                gui;
        ofxLabel                                lblWebcam;
        ofxLabel                                lblSingleMultiple;
        ofxToggle                               togDoTracking;
        ofxLabel                                lblTrackingPersistence;
        ofxLabel                                lblTrackingMaxDistance;
        ofxLabel                                lblDetectorFace, lblDetectorLandmarks;
        ofxLabel                                lblCameraIndex;
        ofxLabel                                lblCameraDimensions;
        ofxLabel                                lblCameraSettings;
        ofxFloatSlider                          sliCertainty;
        ofxIntSlider                            sliKillAfterNotSeenMs;
    
        void callbackCertaintyChanged(float &value);
        void callbackKillAfterChanged(int &value);
    
        // GUI callbacks
        void onDoTrackingChanged(const void* sender, bool& pressed);
    
        appSettings                             settings;
    
        // Some options
        bool                                    bDrawFaces; // draw the faces
        bool                                    bOpenFaceEnabled; // false: disable OpenFace
    
        // A video player
        ofVideoPlayer                           videoPlayer;
        bool                                    bUseVideoFile = false; // true when using a local file instead of the webcam
};
