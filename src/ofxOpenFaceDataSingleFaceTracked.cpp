#include "ofxOpenFaceDataSingleFaceTracked.h"

// Copy all data from the child class.
ofxOpenFaceDataSingleFaceTracked::ofxOpenFaceDataSingleFaceTracked(const ofxOpenFaceDataSingleFace& d) {
    this->detected = d.detected;
    this->gazeLeftEye = d.gazeLeftEye;
    this->gazeRightEye = d.gazeRightEye;
    this->pose = d.pose;
    this->allLandmarks2D = d.allLandmarks2D;
    this->eyeLandmarks2D = d.eyeLandmarks2D;
    this->eyeLandmarks3D = d.eyeLandmarks3D;
    this->certainty = d.certainty;
    this->rBoundingBox = d.rBoundingBox;
    this->sFaceID = d.sFaceID;
}

// Tracker classes
void ofxOpenFaceDataSingleFaceTracked::setup(const ofxOpenFaceDataSingleFace& track) {
    *this = ofxOpenFaceDataSingleFaceTracked(track);
    nTimeAppearedMs = ofGetElapsedTimeMillis();
}

void ofxOpenFaceDataSingleFaceTracked::update(const ofxOpenFaceDataSingleFace& track) {
    *this = ofxOpenFaceDataSingleFaceTracked(track);
}

void ofxOpenFaceDataSingleFaceTracked::kill() {
    dead = true;
}

int ofxOpenFaceDataSingleFaceTracked::getAgeSeconds() const {
    return (ofGetElapsedTimeMillis() - nTimeAppearedMs) / 1000;
}
