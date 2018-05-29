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

void ofxOpenFaceDataSingleFaceTracked::draw() {
    ofxOpenFaceDataSingleFace::draw(true);
    
    if (allLandmarks2D.size() > 0) {
        // Draw label and age
        string s = "Label: " + ofToString(getLabel()) + " / Age: " + ofToString(getAgeSeconds()) + "s";
        // See https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format for landmark indices
        cv::Point cvPt = allLandmarks2D.at(8);
        ofPoint ptChin(cvPt.x, cvPt.y);
        // Draw string
        ofDrawBitmapString(s, ptChin);
    }
}
