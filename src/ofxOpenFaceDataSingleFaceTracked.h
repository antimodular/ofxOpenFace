#include "ofxOpenFaceDataSingleFace.h"

#pragma once

// A class for sharing tracked data for a single face
class ofxOpenFaceDataSingleFaceTracked : public ofxCv::Follower<ofxOpenFaceDataSingleFace>, public ofxOpenFaceDataSingleFace {
public:
    ofxOpenFaceDataSingleFaceTracked() {};
    ofxOpenFaceDataSingleFaceTracked(const ofxOpenFaceDataSingleFace& d); // constructor
    int nTimeAppearedMs = 0; // to keep track of the age
    int nTimeLastSeenMs = 0; // to keep track of the last appearance
    int nTrackingLifeTimeMs = 2000; // time after which we forget a missing face
    
    void setup(const ofxOpenFaceDataSingleFace& track); // called by the tracker when a new face is detected
    void update(const ofxOpenFaceDataSingleFace& track); // called by the tracker when an existing face is updated
    void kill(); // called by the tracker when an existing face is lost
    float getAgeSeconds() const;
    int getLastSeenMs() const; // when were you last seen? milliseconds
    float getLastSeenSecs() const; // when were you last seen? seconds
    void draw(bool bForceDraw = false);
};
