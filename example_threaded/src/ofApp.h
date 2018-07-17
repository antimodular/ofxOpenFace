#pragma once

#define OPENFACE_USE_MULTI false

#include "ofMain.h"
#include "ofxCv.h"
#include "ofxOpenFace.h"


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
    
        ofxOpenFace::Thread oft;
        ofxOpenFace::Tracker tkr;
        ofVideoGrabber grabber;
        bool bRunNewFrameOnly = false;
		
};
