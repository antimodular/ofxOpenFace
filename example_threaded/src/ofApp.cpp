#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    grabber.setup(OPENFACE_WIDTH,OPENFACE_HEIGHT);
    
    oft.lock();
    // manually set model location so we don't need to copy files over
    oft.openFace.model_location = "../../../../../example/bin/data";
    oft.unlock();
    oft.startThread(true); // start the OpenFace thread
}

//--------------------------------------------------------------
void ofApp::update(){
    grabber.update();
}
//--------------------------------------------------------------
void ofApp::exit(){
    oft.stopThread();
}
//--------------------------------------------------------------
void ofApp::draw(){
#if OPENFACE_USE_MULTI
    // MULTI FACE USAGE:
    
    oft.lock(); // lock the thread for reading
    oft.pixels = grabber.getPixels(); // send the camera pixels to openFace
    
    // run on new frames only?
    if (bRunNewFrameOnly){oft.bFrameNew = grabber.isFrameNew();}
    
    // draw camera
    ofSetColor(255); grabber.draw(0,0);
    
    int cnt = 0; // face count
    for (int i = 0; i < oft.openFace.faces.size(); i++){
        
        if (oft.openFace.faces[i].active){ // do something only when the face model is active
            cnt += 1;
            ofColor col = ofColor(255,i*255/(float)oft.openFace.num_faces_max,0);
            oft.openFace.faces[i].draw(col); // draw debug
            
// Obtaining landmarks & other results
//             cv::Vec6d pose        = oft.openFace.faces[i].pose;
//             cv::Point3f gaze0     = oft.openFace.faces[i].gaze0;
//             cv::Point3f gaze1     = oft.openFace.faces[i].gaze1;
//             float certainty       = oft.openFace.faces[i].certainty;
//             cv::Rect_<float> rect = oft.openFace.faces[i].rect;
//             for (int ind = 0; ind < OPENFACE_N_KEYPOINTS; ind ++){
//                 ofVec2f pt = oft.openFace.faces[i].getLandmark(ind);
//             }
            
        }
    }
    
    // send the faces to trackers
    tkr.update(oft.openFace.faces);
    
    // query the face tracker for face labels
    for (int i = 0; i < tkr.tracked_faces.size(); i++){
        ofDrawBitmapStringHighlight("#"+ofToString(tkr.tracked_faces[i].label),
                                                   tkr.tracked_faces[i].currPos.x,
                                                   tkr.tracked_faces[i].currPos.y,
                                    ofColor(0),ofColor(255,255,0));
    }
    
    ofSetColor(255);
    ofxCv::drawMat(oft.openFace.canvas,640,0,320,240);
    ofDrawBitmapStringHighlight("App FPS: "+ofToString(ofGetFrameRate()),0,20,ofColor(0),ofColor(255,255,0));
    ofDrawBitmapStringHighlight("Tracker FPS: "+ofToString(oft.openFace.fps),0,40,ofColor(0),ofColor(255,255,0));
    ofDrawBitmapStringHighlight("Count: "+ofToString(cnt),0,60,ofColor(0),ofColor(255,255,0));
    
    oft.unlock(); // release the lock for OpenFace
#else
    // SINGLE FACE USAGE:
    
    oft.lock(); // lock the thread for reading
    oft.pixels = grabber.getPixels(); // send the camera pixels to openFace
    
    // run on new frames only?
    if (bRunNewFrameOnly){oft.bFrameNew = grabber.isFrameNew();}
    
    // draw camera
    ofSetColor(255); grabber.draw(0,0);
    
    // draw face debug
    oft.openFace.face.draw();
    
    
// Obtaining landmarks & other results
//     cv::Vec6d pose        = oft.openFace.face.pose;
//     cv::Point3f gaze0     = oft.openFace.face.gaze0;
//     cv::Point3f gaze1     = oft.openFace.face.gaze1;
//     float certainty       = oft.openFace.face.certainty;
//     cv::Rect_<float> rect = oft.openFace.face.rect;
//     for (int ind = 0; ind < OPENFACE_N_KEYPOINTS; ind ++){
//         ofVec2f pt = oft.openFace.face.getLandmark(ind);
//     }
    
    ofSetColor(255);
    ofxCv::drawMat(oft.openFace.canvas,640,0,320,240);
    ofDrawBitmapStringHighlight("App FPS: "+ofToString(ofGetFrameRate()),0,20,ofColor(0),ofColor(255,255,0));
    ofDrawBitmapStringHighlight("Tracker FPS: "+ofToString(oft.openFace.fps),0,40,ofColor(0),ofColor(255,255,0));
    ofDrawBitmapStringHighlight("Certainty: "+ofToString(oft.openFace.face.certainty),0,60,ofColor(0),ofColor(255,255,0));
    oft.unlock();
#endif
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    if (key == 'r'){
        oft.lock();
        oft.openFace.reset();
        oft.unlock();
    }
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
