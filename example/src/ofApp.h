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
        int fx, fy, cx, cy; // camera data
        LandmarkDetector::FaceModelParameters::FaceDetector eMethod; // method for tracking
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
        ofxLabel                                lblSingleMultiple;
        ofxLabel                                lblTrackingMethod;
        ofxLabel                                lblTrackingPersistence;
        ofxLabel                                lblTrackingMaxDistance;
        ofxLabel                                lblMultipleMethod;
        ofxLabel                                lblCameraIndex;
        ofxLabel                                lblCameraDimensions;
        ofxLabel                                lblCameraSettings;
    
        appSettings                             settings;
    
        // Some options
        bool                                    bDrawFaces; // draw the faces
        bool                                    bOpenFaceEnabled; // false: disable OpenFace
};
