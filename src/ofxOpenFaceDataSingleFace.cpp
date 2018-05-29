#include "ofxOpenFaceDataSingleFace.h"
#include "ofxOpenFace.h"
#include <VisualizationUtils.h>

void ofxOpenFaceDataSingleFace::draw(bool bForceDraw) {
    if (!detected && !bForceDraw) {
        // Do not draw if no face is detected and draw is not forced
        return;
    }
    
    ofNoFill();
    int nRadius = 2;
    int nThickness = 3;
    
    // Draw the pose
    auto vis_certainty = certainty;
    auto color = cv::Scalar(vis_certainty*255.0, 0, (1 - vis_certainty) * 255);
    auto lines = Utilities::CalculateBox(pose, ofxOpenFace::s_camSettings.fx, ofxOpenFace::s_camSettings.fy, ofxOpenFace::s_camSettings.cx, ofxOpenFace::s_camSettings.cy);
    ofSetLineWidth(nThickness);
    for (size_t i = 0; i < lines.size(); ++i)
    {
        ofPoint pt1 = ofxCv::toOf(lines.at(i).first);
        ofPoint pt2 = ofxCv::toOf(lines.at(i).second);
        ofDrawLine(pt1, pt2);
    }
    
    // Draw all landmarks in 2D
    ofSetColor(ofColor::yellow);
    for (auto pt : allLandmarks2D) {
        ofDrawCircle(pt.x, pt.y, nRadius);
    }
    
    // Draw all eye landmarks in 2D
    ofSetColor(ofColor::blue);
    for (auto pt : eyeLandmarks2D) {
        ofDrawCircle(pt.x, pt.y, nRadius);
    }
    
    // Draw the gazes
    if (eyeLandmarks3D.size() > 0) {
        drawGazes();
    }
    
    if (allLandmarks2D.size() > 0) {
        // Draw the bounding box
        ofRectangle r = ofxCv::toOf(rBoundingBox);
        ofSetColor(ofColor::deepPink);
        ofDrawRectangle(r);
        
        // Draw extra information: ID, certainty
        string s = "ID: " + sFaceID + " / " + ofToString(100.0f * certainty, 0) + "%";
        // See https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format for landmark indices
        cv::Point cvPt = allLandmarks2D.at(33);
        
        ofSetColor(ofColor::yellowGreen);
        ofPoint ptNoseTip(cvPt.x, cvPt.y);
        // Draw string
        ofDrawBitmapString(s, ptNoseTip);
    }
}

void ofxOpenFaceDataSingleFace::drawGazes() {
    int draw_multiplier = 1 << 4;
    // A rough heuristic for drawn point size
    ofSetLineWidth(1);
    
    // First draw the eye region landmarks
    for (size_t i = 0; i < eyeLandmarks2D.size(); ++i)
    {
        cv::Point featurePoint(cvRound(eyeLandmarks2D[i].x * (double)draw_multiplier), cvRound(eyeLandmarks2D[i].y * (double)draw_multiplier));
        
        size_t next_point = i + 1;
        if (i == 7)
            next_point = 0;
        if (i == 19)
            next_point = 8;
        if (i == 27)
            next_point = 20;
        
        if (i == 7 + 28)
            next_point = 0 + 28;
        if (i == 19 + 28)
            next_point = 8 + 28;
        if (i == 27 + 28)
            next_point = 20 + 28;
        
        cv::Point nextFeaturePoint(cvRound(eyeLandmarks2D[next_point].x * (double)draw_multiplier), cvRound(eyeLandmarks2D[next_point].y * (double)draw_multiplier));
        
        ofPoint pt1 = ofxCv::toOf(featurePoint);
        ofPoint pt2 = ofxCv::toOf(nextFeaturePoint);
        
        if ((i < 28 && (i < 8 || i > 19)) || (i >= 28 && (i < 8 + 28 || i > 19 + 28))) {
            // pupil
            ofSetColor(ofColor::mediumPurple);
            ofDrawLine(pt1, pt2);
        } else {
            // eye outlay
            ofSetColor(ofColor::white);
            ofDrawLine(pt1, pt2);
        }
    }
    
    // Now draw the gaze lines themselves
    cv::Mat cameraMat = (cv::Mat_<double>(3, 3) << ofxOpenFace::s_camSettings.fx, 0, ofxOpenFace::s_camSettings.cx, 0, ofxOpenFace::s_camSettings.fy, ofxOpenFace::s_camSettings.cy, 0, 0, 0);
    
    // Grabbing the pupil location, to draw eye gaze need to know where the pupil is
    cv::Point3f pupil_left(0, 0, 0);
    cv::Point3f pupil_right(0, 0, 0);
    for (size_t i = 0; i < 8; ++i)
    {
        pupil_left = pupil_left + eyeLandmarks3D[i];
        pupil_right = pupil_right + eyeLandmarks3D[i + eyeLandmarks3D.size()/2];
    }
    pupil_left = pupil_left / 8;
    pupil_right = pupil_right / 8;
    
    std::vector<cv::Point3d> points_left;
    points_left.push_back(cv::Point3d(pupil_left));
    points_left.push_back(cv::Point3d(pupil_left + cv::Point3f(gazeLeftEye)*50.0));
    
    std::vector<cv::Point3d> points_right;
    points_right.push_back(cv::Point3d(pupil_right));
    points_right.push_back(cv::Point3d(pupil_right + cv::Point3f(gazeRightEye)*50.0));
    
    // TODO: figure out why 3D gaze is not drawn
    cv::Mat_<float> proj_points;
    cv::Mat_<float> mesh_0 = (cv::Mat_<float>(2, 3) << points_left[0].x, points_left[0].y, points_left[0].z, points_left[1].x, points_left[1].y, points_left[1].z);
    Utilities::Project(proj_points, mesh_0, ofxOpenFace::s_camSettings.fx, ofxOpenFace::s_camSettings.fy, ofxOpenFace::s_camSettings.cx, ofxOpenFace::s_camSettings.cy);
    
    ofSetColor(ofColor::aquamarine);
    cv::Point2d cvPt1(cvRound(proj_points.at<double>(0, 0) * (double)draw_multiplier), cvRound(proj_points.at<double>(0, 1) * (double)draw_multiplier));
    cv::Point2d cvPt2(cvRound(proj_points.at<double>(1, 0) * (double)draw_multiplier), cvRound(proj_points.at<double>(1, 1) * (double)draw_multiplier));
    ofPoint pt1 = ofxCv::toOf(cvPt1);
    ofPoint pt2 = ofxCv::toOf(cvPt2);
    ofSetLineWidth(2);
    
    ofDrawLine(pt1, pt2);
    
    cv::Mat_<float> mesh_1 = (cv::Mat_<float>(2, 3) << points_right[0].x, points_right[0].y, points_right[0].z, points_right[1].x, points_right[1].y, points_right[1].z);
    Utilities::Project(proj_points, mesh_1, ofxOpenFace::s_camSettings.fx, ofxOpenFace::s_camSettings.fy, ofxOpenFace::s_camSettings.cx, ofxOpenFace::s_camSettings.cy);
    
    cvPt1 = cv::Point(cvRound(proj_points.at<double>(0, 0) * (double)draw_multiplier), cvRound(proj_points.at<double>(0, 1) * (double)draw_multiplier));
    cvPt2 = cv::Point(cvRound(proj_points.at<double>(1, 0) * (double)draw_multiplier), cvRound(proj_points.at<double>(1, 1) * (double)draw_multiplier));
    ofDrawLine(pt1, pt2);
}
