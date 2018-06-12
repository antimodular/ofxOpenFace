#include "ofxOpenFaceDataSingleFaceTracked.h"
#include "ofxOpenFace.h"

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

void ofxOpenFaceDataSingleFaceTracked::setup(const ofxOpenFaceDataSingleFace& track) {
    *this = ofxOpenFaceDataSingleFaceTracked(track);
    nTimeAppearedMs = ofGetElapsedTimeMillis();
    nTimeLastSeenMs = ofGetElapsedTimeMillis();
}

void ofxOpenFaceDataSingleFaceTracked::update(const ofxOpenFaceDataSingleFace& track) {
    if (track.certainty >= ofxOpenFace::s_fCertaintyNorm) {
        // Remember those values
        auto nTimeAppearedMsPrevious = nTimeAppearedMs;
        auto nTimeLastSeenMsPrevious = nTimeLastSeenMs;
        // Only update time seen if certainty is good enough
        *this = ofxOpenFaceDataSingleFaceTracked(track);
        nTimeAppearedMs = nTimeAppearedMsPrevious; // keep previous value
        // Did it reappear after having disappeared?
        auto timeSinceLastSeenMs = ofGetElapsedTimeMillis() - nTimeLastSeenMsPrevious;
        if (timeSinceLastSeenMs > ofxOpenFace::s_nKillAfterDisappearedMs) {
            // Refresh time appeared
            nTimeAppearedMs = ofGetElapsedTimeMillis();
        }
        nTimeLastSeenMs = ofGetElapsedTimeMillis();
    }
}

void ofxOpenFaceDataSingleFaceTracked::kill() {
    dead = true;
}

float ofxOpenFaceDataSingleFaceTracked::getAgeSeconds() const {
    return (ofGetElapsedTimeMillis() - nTimeAppearedMs) / 1000.0f;
}

int ofxOpenFaceDataSingleFaceTracked::getLastSeenMs() const {
    return ofGetElapsedTimeMillis() - nTimeLastSeenMs;
}

float ofxOpenFaceDataSingleFaceTracked::getLastSeenSecs() const {
    return getLastSeenMs() / 1000.0f;
}

void ofxOpenFaceDataSingleFaceTracked::draw(bool bForceDraw) {
    ofxOpenFaceDataSingleFace::draw(bForceDraw);
    
    if (allLandmarks2D.size() > 0) {
        // Draw label and age
        string s = "Label: " + ofToString(getLabel()) + " / Age: " + ofToString(getAgeSeconds(), 2) + "s / Last seen: " + ofToString(getLastSeenSecs(), 2) + "s";
        // See https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format for landmark indices
        cv::Point cvPt = allLandmarks2D.at(8);
        ofPoint ptChin(cvPt.x, cvPt.y);
        // Draw string
        ofDrawBitmapString(s, ptChin);
    }
}
