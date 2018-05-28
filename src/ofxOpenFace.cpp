#include "ofxOpenFace.h"
#include <Face_utils.h>

ofEvent<ofxOpenFaceDataSingleFace> ofxOpenFace::eventOpenFaceDataSingleRaw = ofEvent<ofxOpenFaceDataSingleFace>();
ofEvent<vector<ofxOpenFaceDataSingleFace>> ofxOpenFace::eventOpenFaceDataMultipleRaw = ofEvent<vector<ofxOpenFaceDataSingleFace>>();
ofEvent<ofxOpenFaceDataSingleFaceTracked> ofxOpenFace::eventOpenFaceDataSingleTracked = ofEvent<ofxOpenFaceDataSingleFaceTracked>();
ofEvent<vector<ofxOpenFaceDataSingleFaceTracked>> ofxOpenFace::eventOpenFaceDataMultipleTracked = ofEvent<vector<ofxOpenFaceDataSingleFaceTracked>>();

ofxOpenFace::CameraSettings ofxOpenFace::s_camSettings;

// Constructor
ofxOpenFace::ofxOpenFace(){
    nMaxFaces = 4; // default value
}

// Destructor
ofxOpenFace::~ofxOpenFace(){
    waitForThread(true);
}

void ofxOpenFace::setup(bool bTrackMultipleFaces, int nWidth, int nHeight, LandmarkDetector::FaceModelParameters::FaceDetector eMethod, CameraSettings settings, int persistenceMs, int maxDistancePx, int nMaxFacesTracked) {
    nImgWidth = nWidth;
    nImgHeight = nHeight;
    bMultipleFaces = bTrackMultipleFaces;
    nMaxFaces = nMaxFacesTracked;
    s_camSettings = settings;
    
    if (bMultipleFaces) {
        setupMultipleFaces(eMethod);
    } else {
        setupSingleFace();
    }
    
    fps_tracker.AddFrame();
    
    // Setup the tracker
    tracker.setPersistence(persistenceMs); // ms before forgetting an object
    tracker.setMaximumDistance(maxDistancePx); // max pixels allowed to move between frames
}

void ofxOpenFace::setupSingleFace() {
    ofFile fModelCLNF = ofFile(OFX_OPENFACE_MODEL);
    pFace_model = new LandmarkDetector::CLNF(fModelCLNF.getAbsolutePath());
    
    if (!pFace_model->eye_model) {
        ofLogError("ofxOpenFace", "No eye model found.");
    }
}

void ofxOpenFace::setupMultipleFaces(LandmarkDetector::FaceModelParameters::FaceDetector eMethod) {
    ofFile fModelCLNF = ofFile(OFX_OPENFACE_MODEL);
    ofFile fDetectorHAAR = ofFile(OFX_OPENFACE_DETECTOR_HAAR);
    ofFile fDetectorMTCNN = ofFile(OFX_OPENFACE_DETECTOR_MTCNN);
    
    if (!fModelCLNF.exists()) {
        ofLogError("ofxOpenFace", "CLNF model doesn't exist at '" + fModelCLNF.getAbsolutePath() + "'");
    }
    if (!fDetectorHAAR.exists()) {
        ofLogError("ofxOpenFace", "HAAR detector doesn't exist at '" + fDetectorHAAR.getAbsolutePath() + "'");
    }
    if (!fDetectorMTCNN.exists()) {
        ofLogError("ofxOpenFace", "MTCNN detector doesn't exist at '" + fDetectorMTCNN.getAbsolutePath() + "'");
    }
    
    // Set up OpenFace
    auto dp = LandmarkDetector::FaceModelParameters();
    dp.curr_face_detector = eMethod;
    if (eMethod == LandmarkDetector::FaceModelParameters::FaceDetector::HOG_SVM_DETECTOR) {
        dp.reinit_video_every = -1;
    }
    vDet_parameters.push_back(dp);
    
#ifdef DO_FACE_ANALYSIS
    // The face analysis logic
    string rootDir = ofFilePath::getAbsolutePath("");
    ofLogNotice("ofxOpenFace", "Face analysis root dir: '" + rootDir + "'");
    pFace_analysis_params = new FaceAnalysis::FaceAnalyserParameters(rootDir);
    pFace_analysis_params->OptimizeForImages();
    pFace_analyser = new FaceAnalysis::FaceAnalyser(*pFace_analysis_params);
    
    ofLogNotice("ofxOpenFace", "Face analysis model location: '" + pFace_analysis_params->getModelLoc() + "'");
    
    if (pFace_analyser->GetAUClassNames().size() == 0 && pFace_analyser->GetAUClassNames().size() == 0)
    {
        ofLogWarning("ofxOpenFace", "No Action Unit models found.");
    }
#endif
    
    // The modules that are being used for tracking
    pFace_model = new LandmarkDetector::CLNF(fModelCLNF.getAbsolutePath()); // the model is always this
    pFace_model->face_detector_HAAR.load(fDetectorHAAR.getAbsolutePath());
    pFace_model->haar_face_detector_location = fDetectorHAAR.getAbsolutePath();
    pFace_model->face_detector_MTCNN.Read(fDetectorMTCNN.getAbsolutePath());
    pFace_model->mtcnn_face_detector_location = fDetectorMTCNN.getAbsolutePath();
    
    if (!pFace_model->loaded_successfully) {
        ofLogError("ofxOpenFace", "The face model was not loaded successfully.");
    }
    
    if (!pFace_model->eye_model) {
        ofLogError("ofxOpenFace", "No eye model found.");
    }
    
    vFace_models.reserve(nMaxFaces);
    vFace_models.push_back(*pFace_model);
    vActiveModels.push_back(false);
    
    for (int i=1; i < nMaxFaces; i++) {
        vFace_models.push_back(*pFace_model);
        vActiveModels.push_back(false);
        vDet_parameters.push_back(dp);
    }
}

ofxOpenFaceDataSingleFace ofxOpenFace::processImageSingleFace() {
    // Reading the images
    mutexImage.lock();
    cv::Mat rgb_image = matToProcessColor;
    cv::Mat grayscale_image = matToProcessGrayScale;
    mutexImage.unlock();
    
    // The actual facial landmark detection / tracking
    ofxOpenFaceDataSingleFace faceData;
    faceData.detected = LandmarkDetector::DetectLandmarksInVideo(rgb_image, *pFace_model, det_parameters, grayscale_image);
     
    // If tracking succeeded and we have an eye model, estimate gaze
    if (faceData.detected && pFace_model->eye_model)
    {
        GazeAnalysis::EstimateGaze(*pFace_model, faceData.gazeLeftEye, s_camSettings.fx, s_camSettings.fy, s_camSettings.cx, s_camSettings.cy, true);
        GazeAnalysis::EstimateGaze(*pFace_model, faceData.gazeRightEye, s_camSettings.fx, s_camSettings.fy, s_camSettings.cx, s_camSettings.cy, false);
    }
    faceData.certainty = pFace_model->detection_certainty;

    // Work out the pose of the head from the tracked model
    faceData.pose = LandmarkDetector::GetPose(*pFace_model, s_camSettings.fx, s_camSettings.fy, s_camSettings.cx, s_camSettings.cy);
    faceData.eyeLandmarks2D = LandmarkDetector::CalculateAllEyeLandmarks(*pFace_model);
    faceData.eyeLandmarks3D = LandmarkDetector::Calculate3DEyeLandmarks(*pFace_model, s_camSettings.fx, s_camSettings.fy, s_camSettings.cx, s_camSettings.cy);
    faceData.allLandmarks2D = LandmarkDetector::CalculateAllLandmarks(*pFace_model);
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

vector<ofxOpenFaceDataSingleFace> ofxOpenFace::processImageMultipleFaces() {
    // Reading the images
    mutexImage.lock();
    cv::Mat rgb_image = matToProcessColor;
    cv::Mat_<uchar> grayscale_image = matToProcessGrayScale;
    mutexImage.unlock();
    
    vector<cv::Rect_<float> > face_detections;
    
    bool all_models_active = true;
    for(unsigned int model = 0; model < vFace_models.size(); ++model) {
        if(!vActiveModels[model]) {
            all_models_active = false;
        }
    }
    
    // Get the detections (every 8th frame and when there are free models available for tracking)
    if(nFrameCount % 8 == 0 && !all_models_active) {
        vector<float> confidences;
        if(vDet_parameters[0].curr_face_detector == LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR) {
            LandmarkDetector::DetectFacesHOG(face_detections, grayscale_image, vFace_models[0].face_detector_HOG, confidences);
        } else if(vDet_parameters[0].curr_face_detector == LandmarkDetector::FaceModelParameters::HAAR_DETECTOR) {
            LandmarkDetector::DetectFaces(face_detections, grayscale_image, vFace_models[0].face_detector_HAAR);
        } else {
            LandmarkDetector::DetectFacesMTCNN(face_detections, grayscale_image, vFace_models[0].face_detector_MTCNN, confidences);
        }
    }
    
    // Keep only non overlapping detections (also convert to a concurrent vector)
    NonOverlapingDetections(vFace_models, face_detections);
    
    vector<tbb::atomic<bool>> face_detections_used(face_detections.size());
    
    vector<ofxOpenFaceDataSingleFace> vData; // the data we will send
    // Initialize it
    for (int i=0; i<nMaxFaces; i++) {
        ofxOpenFaceDataSingleFace d;
        vData.push_back(d);
    }
    
    // Go through every model and update the tracking
    //tbb::parallel_for(0, (int)vFace_models.size(), [&](int model) {
    for (unsigned int model = 0; model < vFace_models.size(); ++model) {
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
                    detection_success = LandmarkDetector::DetectLandmarksInVideo(rgb_image, face_detections[detection_ind], vFace_models[model], vDet_parameters[model], grayscale_image);
                    
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
            detection_success = LandmarkDetector::DetectLandmarksInVideo(rgb_image, vFace_models[model], vDet_parameters[model], grayscale_image);
        }
        
        vData[model].detected = detection_success;
        vData[model].certainty = vFace_models[model].detection_certainty;
        vData[model].pose = LandmarkDetector::GetPose(vFace_models[model], s_camSettings.fx, s_camSettings.fy, s_camSettings.cx, s_camSettings.cy);
        vData[model].eyeLandmarks2D = LandmarkDetector::CalculateAllEyeLandmarks(vFace_models[model]);
        vData[model].eyeLandmarks3D = LandmarkDetector::Calculate3DEyeLandmarks(vFace_models[model], s_camSettings.fx, s_camSettings.fy, s_camSettings.cx, s_camSettings.cy);
        vData[model].allLandmarks2D = LandmarkDetector::CalculateAllLandmarks(vFace_models[model]);
        vData[model].sFaceID = ofToString(model + 1);
        GazeAnalysis::EstimateGaze(vFace_models[model], vData[model].gazeLeftEye, s_camSettings.fx, s_camSettings.fy, s_camSettings.cx, s_camSettings.cy, true);
        GazeAnalysis::EstimateGaze(vFace_models[model], vData[model].gazeRightEye, s_camSettings.fx, s_camSettings.fy, s_camSettings.cx, s_camSettings.cy, false);
        
        // Figure out the bounding box of all landmarks
        vector<ofPoint> vLandmarks2D;
        for (auto& l : vData[model].allLandmarks2D) {
            vLandmarks2D.push_back(ofPoint(l.x, l.y));
        }
        ofPolyline pl;
        pl.addVertices(vLandmarks2D);
        pl.close();
        vData[model].rBoundingBox = ofxCv::toCv(pl.getBoundingBox());
    }
    //});
    
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
        pFace_model->Reset();
    }
}

void ofxOpenFace::threadedFunction() {
    thread.setName("ofxOpenFace " + thread.name());
    ofLogNotice("ofxOpenFace", "Thread started.");
    
    while(!bExit) {
        // Do we have an image to process?
        if (!bHaveNewImage) {
            ofSleepMillis(20);
        } else {
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
                std::vector<ofxOpenFaceDataSingleFace> v;
                v.push_back(d);
                tracker.track(v);
                // Raise the event for the updated faces
                ofNotifyEvent(eventOpenFaceDataSingleRaw, d);
                // Raise the event for the tracked faces
                if (tracker.getFollowers().size() > 0) {
                    ofNotifyEvent(eventOpenFaceDataSingleTracked, tracker.getFollowers().front());
                }
            }
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

void ofxOpenFace::NonOverlapingDetections(const vector<LandmarkDetector::CLNF>& clnf_models, vector<cv::Rect_<float> >& face_detections) {
    // Go over the model and eliminate detections that are not informative (there already is a tracker there)
    for(size_t model = 0; model < clnf_models.size(); ++model)
    {        
        // See if the detections intersect
        cv::Rect_<float> model_rect = clnf_models[model].GetBoundingBox();
        
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

vector<ofxOpenFaceDataSingleFaceTracked> ofxOpenFace::getTracked() {
    return tracker.getFollowers();
}

// Return the tracking distance between two objects
float ofxCv::trackingDistance(const ofxOpenFaceDataSingleFaceTracked& a, const ofxOpenFaceDataSingleFaceTracked& b) {
    // For now, use the tracking distance of the bounding boxes
    return ofxCv::trackingDistance(a.rBoundingBox, b.rBoundingBox);
}
