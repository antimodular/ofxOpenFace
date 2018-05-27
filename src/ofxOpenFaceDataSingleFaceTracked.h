#include "ofxOpenFaceDataSingleFace.h"

#pragma once

// A class for sharing tracked data for a single face
class ofxOpenFaceDataSingleFaceTracked : public ofxCv::Follower<ofxOpenFaceDataSingleFace>, public ofxOpenFaceDataSingleFace {
public:
    ofxOpenFaceDataSingleFaceTracked() {};
    ofxOpenFaceDataSingleFaceTracked(const ofxOpenFaceDataSingleFace& d); // constructor
    int nTimeAppearedMs = 0; // to keep track of the age
    
    void setup(const ofxOpenFaceDataSingleFace& track); // called by the tracker when a new face is detected
    void update(const ofxOpenFaceDataSingleFace& track); // called by the tracker when an existing face is updated
    void kill(); // called by the tracker when an existing face is lost
    int getAgeSeconds() const;
    // int getLastSeenMs() const; // when were you last seen?
};
