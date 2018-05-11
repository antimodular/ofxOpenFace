#include "ofxOpenFace.h"
#include <Face_utils.h>

ofEvent<OpenFaceDataSingleFace> ofxOpenFace::eventOpenFaceDataSingleRaw = ofEvent<OpenFaceDataSingleFace>();
ofEvent<vector<OpenFaceDataSingleFace>> ofxOpenFace::eventOpenFaceDataMultipleRaw = ofEvent<vector<OpenFaceDataSingleFace>>();
ofEvent<OpenFaceDataSingleFaceTracked> ofxOpenFace::eventOpenFaceDataSingleTracked = ofEvent<OpenFaceDataSingleFaceTracked>();
ofEvent<vector<OpenFaceDataSingleFaceTracked>> ofxOpenFace::eventOpenFaceDataMultipleTracked = ofEvent<vector<OpenFaceDataSingleFaceTracked>>();

// Constructor
ofxOpenFace::ofxOpenFace(){
    nMaxFaces = 4; // default value
}

// Destructor
ofxOpenFace::~ofxOpenFace(){
    waitForThread(true);
}

void ofxOpenFace::setup(bool bTrackMultipleFaces, int nWidth, int nHeight, bool bUseHOGSVM, int persistenceMs, int maxDistancePx, int nMaxFacesTracked) {
    nImgWidth = nWidth;
    nImgHeight = nHeight;
    bMultipleFaces = bTrackMultipleFaces;
    nMaxFaces = nMaxFacesTracked;
    
    // Initialize some parameters. See https://github.com/TadasBaltrusaitis/OpenFace/wiki/API-calls
    fx = 500.0f;
    fy = 500.0f;
    cx = (float)nImgWidth/2.0f;
    cy = (float)nImgHeight/2.0f;
    
    if (bMultipleFaces) {
        setupMultipleFaces(bUseHOGSVM);
    } else {
        setupSingleFace();
    }
    
    fps_tracker.AddFrame();
    
    // Setup the tracker
    tracker.setPersistence(persistenceMs); // ms before forgetting an object
    tracker.setMaximumDistance(maxDistancePx); // max pixels allowed to move between frames
}

void ofxOpenFace::setupSingleFace() {
    string modelLocation = ofFilePath::getAbsolutePath("model/main_clnf_general.txt");
    face_model = LandmarkDetector::CLNF(modelLocation);
    
    if (!face_model.eye_model) {
        ofLogError("ofxOpenFace", "No eye model found.");
    }
}

void ofxOpenFace::setupMultipleFaces(bool bUseHOGSVM) {
    string modelLocation = ofFilePath::getAbsolutePath("model/main_clnf_general.txt");
    string detectorLocationHAAR = ofFilePath::getAbsolutePath("classifiers/haarcascade_frontalface_alt.xml");
    
    // Set up OpenFace
    auto dp = LandmarkDetector::FaceModelParameters();
    if (bUseHOGSVM) {
        dp.curr_face_detector = LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR;
        dp.reinit_video_every = -1;
    } else {
        dp.curr_face_detector = LandmarkDetector::FaceModelParameters::HAAR_DETECTOR;
    }
    vDet_parameters.push_back(dp);
    
    // The modules that are being used for tracking
    face_model = LandmarkDetector::CLNF(modelLocation);
    if (!bUseHOGSVM) {
        face_model.face_detector_HAAR.load(detectorLocationHAAR);
        face_model.face_detector_location = detectorLocationHAAR;
    }
    
    vFace_models.reserve(nMaxFaces);
    vFace_models.push_back(face_model);
    vActiveModels.push_back(false);
    
    for (int i=1; i < nMaxFaces; i++) {
        vFace_models.push_back(face_model);
        vActiveModels.push_back(false);
        vDet_parameters.push_back(dp);
    }
    
    if (!face_model.eye_model) {
        ofLogError("ofxOpenFace", "No eye model found.");
    }
}

OpenFaceDataSingleFace ofxOpenFace::processImageSingleFace() {
    // Reading the images
    cv::Mat captured_image = matToProcessColor;
    cv::Mat grayscale_image = matToProcessGrayScale;
    
    // The actual facial landmark detection / tracking
    OpenFaceDataSingleFace faceData;
    faceData.detected = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, face_model, det_parameters);
     
    // If tracking succeeded and we have an eye model, estimate gaze
    if (faceData.detected && face_model.eye_model)
    {
        GazeAnalysis::EstimateGaze(face_model, faceData.gazeLeftEye, fx, fy, cx, cy, true);
        GazeAnalysis::EstimateGaze(face_model, faceData.gazeRightEye , fx, fy, cx, cy, false);
    }
    faceData.certainty = face_model.detection_certainty;

    // Work out the pose of the head from the tracked model
    faceData.pose = LandmarkDetector::GetPose(face_model, fx, fy, cx, cy);
    faceData.eyeLandmarks2D = LandmarkDetector::CalculateAllEyeLandmarks(face_model);
    faceData.eyeLandmarks3D = LandmarkDetector::Calculate3DEyeLandmarks(face_model, fx, fy, cx, cy);
    faceData.allLandmarks2D = LandmarkDetector::CalculateAllLandmarks(face_model);
    faceData.sFaceID = ofToString(1);
    
    // Figure out the bounding box of all landmarks
    vector<ofPoint> vLandmarks2D;
    for (auto& l : faceData.allLandmarks2D) {
        vLandmarks2D.push_back(ofPoint(l.x, l.y));
    }
    ofPolyline pl;
    pl.addVertices(vLandmarks2D);
    pl.close();
    faceData.rBoundingBox = ofxCv::toCv(pl.getBoundingBox());
    
    return faceData;
}

vector<OpenFaceDataSingleFace> ofxOpenFace::processImageMultipleFaces() {
    // Reading the images
    cv::Mat captured_image = matToProcessColor;
    cv::Mat grayscale_image = matToProcessGrayScale;
    
    vector<cv::Rect_<double> > face_detections;
    
    bool all_models_active = true;
    for(unsigned int model = 0; model < vFace_models.size(); ++model)
    {
        if(!vActiveModels[model])
        {
            all_models_active = false;
        }
    }
    
    // Get the detections (every 8th frame and when there are free models available for tracking)
    if(nFrameCount % 8 == 0 && !all_models_active)
    {
        if(vDet_parameters[0].curr_face_detector == LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR)
        {
            vector<double> confidences;
            LandmarkDetector::DetectFacesHOG(face_detections, grayscale_image, vFace_models[0].face_detector_HOG, confidences);
        }
        else
        {
            LandmarkDetector::DetectFaces(face_detections, grayscale_image, vFace_models[0].face_detector_HAAR);
        }
    }
    
    // Keep only non overlapping detections (also convert to a concurrent vector
    NonOverlapingDetections(vFace_models, face_detections);
    
    vector<tbb::atomic<bool>> face_detections_used(face_detections.size());
    
    vector<OpenFaceDataSingleFace> vData; // the data we will send
    for (int i=0; i<nMaxFaces; i++) {
        OpenFaceDataSingleFace d;
        vData.push_back(d);
    }
    
    // Go through every model and update the tracking
    tbb::parallel_for(0, (int)vFace_models.size(), [&](int model) {
        bool detection_success = false;
        
        // If the current model has failed more than 4 times in a row, remove it
        if(vFace_models[model].failures_in_a_row > 4)
        {
            vActiveModels[model] = false;
            vFace_models[model].Reset();
        }
        
        // If the model is inactive reactivate it with new detections
        if(!vActiveModels[model])
        {
            for(size_t detection_ind = 0; detection_ind < face_detections.size(); ++detection_ind)
            {
                // if it was not taken by another tracker take it (if it is false swap it to true and enter detection, this makes it parallel safe)
                if(face_detections_used[detection_ind].compare_and_swap(true, false) == false)
                {
                    // Reinitialise the model
                    vFace_models[model].Reset();
                    
                    // This ensures that a wider window is used for the initial landmark localisation
                    vFace_models[model].detection_success = false;
                    detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, face_detections[detection_ind], vFace_models[model], vDet_parameters[model]);
                    
                    // This activates the model
                    vActiveModels[model] = true;
                    
                    // break out of the loop as the tracker has been reinitialised
                    break;
                }
            }
        }
        else
        {
            // The actual facial landmark detection / tracking
            detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, vFace_models[model], vDet_parameters[model]);
        }
        
        vData[model].detected = detection_success;
        vData[model].certainty = vFace_models[model].detection_certainty;
        vData[model].pose = LandmarkDetector::GetPose(vFace_models[model], fx, fy, cx, cy);
        vData[model].eyeLandmarks2D = LandmarkDetector::CalculateAllEyeLandmarks(vFace_models[model]);
        vData[model].eyeLandmarks3D = LandmarkDetector::Calculate3DEyeLandmarks(vFace_models[model], fx, fy, cx, cy);
        vData[model].allLandmarks2D = LandmarkDetector::CalculateAllLandmarks(vFace_models[model]);
        vData[model].sFaceID = ofToString(model + 1);
        GazeAnalysis::EstimateGaze(vFace_models[model], vData[model].gazeLeftEye, fx, fy, cx, cy, true);
        GazeAnalysis::EstimateGaze(vFace_models[model], vData[model].gazeRightEye, fx, fy, cx, cy, false);
        
        // Figure out the bounding box of all landmarks
        vector<ofPoint> vLandmarks2D;
        for (auto& l : vData[model].allLandmarks2D) {
            vLandmarks2D.push_back(ofPoint(l.x, l.y));
        }
        ofPolyline pl;
        pl.addVertices(vLandmarks2D);
        pl.close();
        vData[model].rBoundingBox = ofxCv::toCv(pl.getBoundingBox());
    });
    
    // Update the frame count
    nFrameCount++;
    
    return vData;
}

void ofxOpenFace::setImage(ofImage img) {
    mutexImage.lock();
    // Override the current "next image"
    matToProcessColor = ofxCv::toCv(img.getPixels());
    img.setImageType(ofImageType::OF_IMAGE_GRAYSCALE);
    matToProcessGrayScale = ofxCv::toCv(img.getPixels());
    bHaveNewImage = true;
    mutexImage.unlock();
}

void ofxOpenFace::stop() {
    bExit = true;
    ofLogNotice("ofxOpenFace", "Stopping thread.");
}

void ofxOpenFace::exit() {
    // Clear memory
    stop();
    waitForThread(true);
}

void ofxOpenFace::resetFaceModel() {
    if (bMultipleFaces) {
        for(size_t i=0; i < vFace_models.size(); ++i)
        {
            vFace_models[i].Reset();
            vActiveModels[i] = false;
        }
    } else {
        face_model.Reset();
    }
}

void ofxOpenFace::threadedFunction() {
    thread.setName("ofxOpenFace " + thread.name());
    ofLogNotice("ofxOpenFace", "Thread started.");
    
    while(!bExit) {
        float fSmoothing = 0.9f;
        int timeBeforeMs = ofGetElapsedTimeMillis();
        
        // Do we have an image to process?
        if (!bHaveNewImage) {
            ofSleepMillis(20);
        } else {
            //ofLogNotice("ofxOpenFace", "New image to process.");
            mutexImage.lock();
            nFrameCount = 0;
            if (bMultipleFaces) {
                auto v = processImageMultipleFaces();
                // Update the tracker
                tracker.track(v);
                // Raise the event for the updated faces
                ofNotifyEvent(eventOpenFaceDataMultipleRaw, v);
                // Raise the event for the tracked faces
                ofNotifyEvent(eventOpenFaceDataMultipleTracked, tracker.getFollowers());
            } else {
                auto d = processImageSingleFace();
                // Update the tracker
                std::vector<OpenFaceDataSingleFace> v;
                v.push_back(d);
                tracker.track(v);
                // Raise the event for the updated faces
                ofNotifyEvent(eventOpenFaceDataSingleRaw, d);
                // Raise the event for the tracked faces
                if (tracker.getFollowers().size() > 0) {
                    ofNotifyEvent(eventOpenFaceDataSingleTracked, tracker.getFollowers().front());
                }
            }
            mutexImage.unlock();
            bHaveNewImage = false; // ready for a new image
        }
        fps_tracker.AddFrame();
    }
    bHaveNewImage = false;
}

int ofxOpenFace::getFPS() {
    int nToReturn = (int)fps_tracker.GetFPS();
    return nToReturn;
}

void ofxOpenFace::NonOverlapingDetections(const vector<LandmarkDetector::CLNF>& clnf_models, vector<cv::Rect_<double> >& face_detections) {
    // Go over the model and eliminate detections that are not informative (there already is a tracker there)
    for(size_t model = 0; model < clnf_models.size(); ++model)
    {        
        // See if the detections intersect
        cv::Rect_<double> model_rect = clnf_models[model].GetBoundingBox();
        
        for(int detection = face_detections.size()-1; detection >=0; --detection)
        {
            double intersection_area = (model_rect & face_detections[detection]).area();
            double union_area = model_rect.area() + face_detections[detection].area() - 2 * intersection_area;
            
            // If the model is already tracking what we're detecting ignore the detection, this is determined by amount of overlap
            if( intersection_area/union_area > 0.5)
            {
                face_detections.erase(face_detections.begin() + detection);
            }
        }
    }
}

// Draw the face data in the referenced material
void ofxOpenFace::drawFaceIntoMaterial(cv::Mat& mat, const OpenFaceDataSingleFace& data, bool bForceDraw) {
    if (!data.detected && !bForceDraw) {
        // Do not draw if no face is detected and draw is not forced
        return;
    }
    
    ofNoFill();
    
    // Draw the pose
    auto vis_certainty = data.certainty;
    auto color = cv::Scalar(vis_certainty*255.0, 0, (1 - vis_certainty) * 255);
    Utilities::DrawBox(mat, data.pose, color, 3, fx, fy, cx, cy);
    
    // Draw all landmarks in 2D
    int nRadius = 2;
    int nThickness = 1;
    color = ofxCv::toCv(ofColor::yellow);
    for (auto pt : data.allLandmarks2D) {
        cv::circle(mat, cv::Point(pt.x, pt.y), 2, color);
    }
    
    // Draw all eye landmarks in 2D
    color = ofxCv::toCv(ofColor::blue);
    for (auto pt : data.eyeLandmarks2D) {
        cv::circle(mat, cv::Point(pt.x, pt.y), 2, color);
    }
    
    // Draw the gazes
    drawGazes(mat, data);
    
    if (data.allLandmarks2D.size() > 0) {
        // Draw the bounding box
        ofRectangle r = ofxCv::toOf(data.rBoundingBox);
        cv::rectangle(mat, cv::Point(r.getTopLeft().x, r.getTopLeft().y),
                      cv::Point(r.getBottomRight().x, r.getBottomRight().y), ofxCv::toCv(ofColor::deepPink));
        
        // Draw extra information: ID, certainty
        string s = "ID: " + data.sFaceID + " / " + ofToString(100.0f * data.certainty, 0) + "%";
        // See https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format for landmark indices
        cv::Point ptNoseTip = data.allLandmarks2D.at(33);
        cv::putText(mat, s, ptNoseTip, 0, 1.0, ofxCv::toCv(ofColor::yellowGreen), 3);
    }
}

// Draw the face data in the referenced material
void ofxOpenFace::drawTrackedFaceIntoMaterial(cv::Mat& mat, const OpenFaceDataSingleFaceTracked& data) {
    drawFaceIntoMaterial(mat, (const OpenFaceDataSingleFace&)data);
    
    if (data.detected && data.allLandmarks2D.size() > 0) {
        // Draw label and age
        string s = "Label: " + ofToString(data.getLabel()) + " / Age: " + ofToString(data.getAgeSeconds()) + "s";
        // See https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format for landmark indices
        cv::Point ptChin = data.allLandmarks2D.at(8);
        cv::putText(mat, s, ptChin, 0, 1.0, ofxCv::toCv(ofColor::darkGreen), 3);
    }
}

void ofxOpenFace::drawGazes(cv::Mat& mat, const OpenFaceDataSingleFace& data) {
    // First draw the eye region landmarks
    for (size_t i = 0; i < data.eyeLandmarks2D.size(); ++i)
    {
        cv::Point featurePoint(cvRound(data.eyeLandmarks2D[i].x * (double)draw_multiplier), cvRound(data.eyeLandmarks2D[i].y * (double)draw_multiplier));
        
        // A rough heuristic for drawn point size
        int thickness = 1;
        int thickness_2 = 1;
        
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
        
        cv::Point nextFeaturePoint(cvRound(data.eyeLandmarks2D[next_point].x * (double)draw_multiplier), cvRound(data.eyeLandmarks2D[next_point].y * (double)draw_multiplier));
        if ((i < 28 && (i < 8 || i > 19)) || (i >= 28 && (i < 8 + 28 || i > 19 + 28))) {
            // pupil
            cv::line(mat, featurePoint, nextFeaturePoint, ofxCv::toCv(ofColor::mediumPurple), thickness_2, CV_AA, draw_shiftbits);
        } else {
            // eye outlay
            cv::line(mat, featurePoint, nextFeaturePoint, ofxCv::toCv(ofColor::white), thickness_2, CV_AA, draw_shiftbits);
        }
        
    }
    
    // Now draw the gaze lines themselves
    cv::Mat cameraMat = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 0);
    
    // Grabbing the pupil location, to draw eye gaze need to know where the pupil is
    cv::Point3d pupil_left(0, 0, 0);
    cv::Point3d pupil_right(0, 0, 0);
    for (size_t i = 0; i < 8; ++i)
    {
        pupil_left = pupil_left + data.eyeLandmarks3D[i];
        pupil_right = pupil_right + data.eyeLandmarks3D[i + data.eyeLandmarks3D.size()/2];
    }
    pupil_left = pupil_left / 8;
    pupil_right = pupil_right / 8;
    
    std::vector<cv::Point3d> points_left;
    points_left.push_back(cv::Point3d(pupil_left));
    points_left.push_back(cv::Point3d(pupil_left + cv::Point3d(data.gazeLeftEye)*50.0));
    
    std::vector<cv::Point3d> points_right;
    points_right.push_back(cv::Point3d(pupil_right));
    points_right.push_back(cv::Point3d(pupil_right + cv::Point3d(data.gazeRightEye)*50.0));
    
    // TODO: figure out why 3D gaze is not drawn
    cv::Mat_<float> proj_points;
    cv::Mat_<float> mesh_0 = (cv::Mat_<float>(2, 3) << points_left[0].x, points_left[0].y, points_left[0].z, points_left[1].x, points_left[1].y, points_left[1].z);
    FaceAnalysis::Project(proj_points, mesh_0, fx, fy, cx, cy);
    cv::line(mat, cv::Point(cvRound(proj_points.at<double>(0, 0) * (double)draw_multiplier), cvRound(proj_points.at<double>(0, 1) * (double)draw_multiplier)),
             cv::Point(cvRound(proj_points.at<double>(1, 0) * (double)draw_multiplier), cvRound(proj_points.at<double>(1, 1) * (double)draw_multiplier)), cv::Scalar(110, 220, 0), 2, CV_AA, draw_shiftbits);
    
    cv::Mat_<float> mesh_1 = (cv::Mat_<float>(2, 3) << points_right[0].x, points_right[0].y, points_right[0].z, points_right[1].x, points_right[1].y, points_right[1].z);
    FaceAnalysis::Project(proj_points, mesh_1, fx, fy, cx, cy);
    cv::line(mat, cv::Point(cvRound(proj_points.at<double>(0, 0) * (double)draw_multiplier), cvRound(proj_points.at<double>(0, 1) * (double)draw_multiplier)),
             cv::Point(cvRound(proj_points.at<double>(1, 0) * (double)draw_multiplier), cvRound(proj_points.at<double>(1, 1) * (double)draw_multiplier)), cv::Scalar(110, 220, 0), 2, CV_AA, draw_shiftbits);

}

// Draw all tracked faces in one function
void ofxOpenFace::drawTrackedIntoMaterial(cv::Mat& mat) {
    vector<OpenFaceDataSingleFaceTracked> followed = tracker.getFollowers();
    for (auto &f : followed) {
        // Draw the face
        drawTrackedFaceIntoMaterial(mat, f);
    }
}

// Copy all data from the child class.
OpenFaceDataSingleFaceTracked::OpenFaceDataSingleFaceTracked(const OpenFaceDataSingleFace& d) {
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
void OpenFaceDataSingleFaceTracked::setup(const OpenFaceDataSingleFace& track) {
    *this = OpenFaceDataSingleFaceTracked(track);
    nTimeAppearedMs = ofGetElapsedTimeMillis();
}

void OpenFaceDataSingleFaceTracked::update(const OpenFaceDataSingleFace& track) {
    *this = OpenFaceDataSingleFaceTracked(track);
}

void OpenFaceDataSingleFaceTracked::kill() {
    dead = true;
}

int OpenFaceDataSingleFaceTracked::getAgeSeconds() const {
    return (ofGetElapsedTimeMillis() - nTimeAppearedMs) / 1000;
}

// Return the tracking distance between two objects
float ofxCv::trackingDistance(const OpenFaceDataSingleFaceTracked& a, const OpenFaceDataSingleFaceTracked& b) {
    // For now, use the tracking distance of the bounding boxes
    return ofxCv::trackingDistance(a.rBoundingBox, b.rBoundingBox);
}
