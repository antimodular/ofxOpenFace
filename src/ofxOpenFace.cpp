#include "ofxOpenFace.h"

ofEvent<OpenFaceDataSingleFace> ofxOpenFace::eventDataReadySingleFace = ofEvent<OpenFaceDataSingleFace>();

ofEvent<OpenFaceDataMultipleFaces> ofxOpenFace::eventDataReadyMultipleFaces = ofEvent<OpenFaceDataMultipleFaces>();

// Constructor
ofxOpenFace::ofxOpenFace(){
    nMaxFaces = 4;
}

// Destructor
ofxOpenFace::~ofxOpenFace(){
    waitForThread(true);
}

void ofxOpenFace::setup(bool bTrackMultipleFaces, int nWidth, int nHeight) {
    nImgWidth = nWidth;
    nImgHeight = nHeight;
    
    bMultipleFaces = bTrackMultipleFaces;
    
    // Initialize some parameters. See https://github.com/TadasBaltrusaitis/OpenFace/wiki/API-calls
    fx = 500.0f;
    fy = 500.0f;
    cx = (float)nImgWidth/2.0f;
    cy = (float)nImgHeight/2.0f;
    
    if (bMultipleFaces) {
        setupMultipleFaces();
    } else {
        setupSingleFace();
    }
}

void ofxOpenFace::setupSingleFace() {
    // Set up OpenFace
    vector<string> arguments;
    arguments.push_back("-device");
    arguments.push_back("0");
    pDet_parameters = new LandmarkDetector::FaceModelParameters(arguments);
    
    // The modules that are being used for tracking
    string modelLocation = ofFilePath::getAbsolutePath("model/main_clnf_general.txt");
    pFace_model = new LandmarkDetector::CLNF(modelLocation);
    
    if (!pFace_model->eye_model) {
        ofLogError("ofxOpenFace", "No eye model found.");
    }
    
    // A utility for visualizing the results
    pVisualizer = new Utilities::Visualizer(true, false, false, false);
    bDoVisualizer = false;
}

void ofxOpenFace::setupMultipleFaces() {
    // Set up OpenFace
    string sAppRunDir = "/Users/bruno/DEV/of_v0.9.8_osx_release/apps/myApps/openFace/bin/data/dummy.txt";
    vector<string> arguments;
    arguments.push_back(sAppRunDir); // so that the FaceAnalyserParameters constructor can find the models in data/AU_predictors
    arguments.push_back("-device");
    arguments.push_back("0"); // not used because the image is sent via setImage
    auto det_params = LandmarkDetector::FaceModelParameters(arguments);
    
    // This is so that the model would not try re-initialising itself
    det_params.reinit_video_every = -1;
    det_params.curr_face_detector = LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR;
    vDet_parameters.push_back(det_params);
    
    // The modules that are being used for tracking
    string modelLocation = ofFilePath::getAbsolutePath("model/main_clnf_general.txt");
    string detectorLocation = ofFilePath::getAbsolutePath("classifiers/haarcascade_frontalface_alt.xml");
    LandmarkDetector::CLNF face_model = LandmarkDetector::CLNF(modelLocation);
    face_model.face_detector_HAAR.load(detectorLocation);
    face_model.face_detector_location = detectorLocation;
    
    vFace_models.reserve(nMaxFaces);
    vFace_models.push_back(face_model);
    vActiveModels.push_back(false);
    
    for (int i=1; i < nMaxFaces; i++) {
        vFace_models.push_back(face_model);
        vActiveModels.push_back(false);
        vDet_parameters.push_back(det_params);
    }
    
    // Load facial feature extractor and AU analyser (make sure it is static, as we don't reidentify faces)
    pFace_analysis_params = new FaceAnalysis::FaceAnalyserParameters(arguments);
    pFace_analysis_params->OptimizeForVideos();
    ofLogNotice("ofxOpenFace", "FaceAnalyserParameters model location: '" + pFace_analysis_params->getModelLoc() + "'");
    pFace_analyser = new FaceAnalysis::FaceAnalyser(*pFace_analysis_params);
    
    if (!face_model.eye_model) {
        ofLogError("ofxOpenFace", "No eye model found.");
    }
    
    if (pFace_analyser->GetAUClassNames().size() == 0 && pFace_analyser->GetAUClassNames().size() == 0)
    {
        ofLogError("ofxOpenFace", "No Action Unit models found.");
    }
    
    // A utility for visualizing the results
    pVisualizer = new Utilities::Visualizer(true, false, false, false);
    bDoVisualizer = false;
}

void ofxOpenFace::processImageSingleFace() {
    // Reading the images
    cv::Mat captured_image = matToProcessColor;
    cv::Mat grayscale_image = matToProcessGrayScale;
    
    // The actual facial landmark detection / tracking
    OpenFaceDataSingleFace faceData;
    faceData.detected = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, *pFace_model, *pDet_parameters);
     
    // If tracking succeeded and we have an eye model, estimate gaze
    if (faceData.detected && pFace_model->eye_model)
    {
        GazeAnalysis::EstimateGaze(*pFace_model, faceData.gazeLeftEye, fx, fy, cx, cy, true);
        GazeAnalysis::EstimateGaze(*pFace_model, faceData.gazeRightEye , fx, fy, cx, cy, false);
    }
    faceData.certainty = pFace_model->detection_certainty;

    // Work out the pose of the head from the tracked model
    faceData.pose = LandmarkDetector::GetPose(*pFace_model, fx, fy, cx, cy);
    faceData.eyeLandmarks2D = LandmarkDetector::CalculateAllEyeLandmarks(*pFace_model);
    faceData.eyeLandmarks3D = LandmarkDetector::Calculate3DEyeLandmarks(*pFace_model, fx, fy, cx, cy);
    faceData.allLandmarks2D = LandmarkDetector::CalculateAllLandmarks(*pFace_model);

    if (bDoVisualizer) {
        // Displaying the tracking visualizations
        pVisualizer->SetImage(captured_image, fx, fy, cx, cy);
        pVisualizer->SetObservationLandmarks(pFace_model->detected_landmarks, faceData.certainty, pFace_model->GetVisibilities());
        pVisualizer->SetObservationPose(faceData.pose, faceData.certainty);
        pVisualizer->SetObservationGaze(faceData.gazeLeftEye, faceData.gazeRightEye, faceData.eyeLandmarks2D, faceData.eyeLandmarks3D, faceData.certainty);
    }
    ofNotifyEvent(eventDataReadySingleFace, faceData);
}

void ofxOpenFace::processImageMultipleFaces() {
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
    
    vector<tbb::atomic<bool> > face_detections_used(face_detections.size());
    
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
        
        if (detection_success) {
            ofLogNotice("ofxOpenFace", "Detection successful");
        }
        
        vData[model].detected = detection_success;
        vData[model].certainty = vFace_models[model].detection_certainty;
        vData[model].pose = LandmarkDetector::GetPose(vFace_models[model], fx, fy, cx, cy);
        vData[model].eyeLandmarks2D = LandmarkDetector::CalculateAllEyeLandmarks(vFace_models[model]);
        vData[model].eyeLandmarks3D = LandmarkDetector::Calculate3DEyeLandmarks(vFace_models[model], fx, fy, cx, cy);
        vData[model].allLandmarks2D = LandmarkDetector::CalculateAllLandmarks(vFace_models[model]);
    });
    
    // Raise the event for the updated faces
    OpenFaceDataMultipleFaces faceData;
    faceData.vFaces = vData;
    ofNotifyEvent(eventDataReadyMultipleFaces, faceData);
    
    if (bDoVisualizer) {
        pVisualizer->SetImage(captured_image, fx, fy, cx, cy);
        auto face_model = vFace_models[0];
    
        // Go through every model and detect eye gaze, record results and visualise the results
        for(size_t model = 0; model < vFace_models.size(); ++model)
        {
            // Visualising the results
            if(vActiveModels[model])
            {
                
                // Estimate head pose and eye gaze
                cv::Vec6d pose_estimate = LandmarkDetector::GetPose(vFace_models[model], fx, fy, cx, cy);
                
                cv::Point3f gaze_direction0(0, 0, 0); cv::Point3f gaze_direction1(0, 0, 0); cv::Vec2d gaze_angle(0, 0);
                
                // Detect eye gazes
                if (vFace_models[model].detection_success && face_model.eye_model)
                {
                    GazeAnalysis::EstimateGaze(vFace_models[model], gaze_direction0, fx, fy, cx, cy, true);
                    GazeAnalysis::EstimateGaze(vFace_models[model], gaze_direction1, fx, fy, cx, cy, false);
                    gaze_angle = GazeAnalysis::GetGazeAngle(gaze_direction0, gaze_direction1);
                }
                
                // Face analysis step
                cv::Mat sim_warped_img;
                cv::Mat_<double> hog_descriptor; int num_hog_rows = 0, num_hog_cols = 0;
                
                /*
                // Perform AU detection and HOG feature extraction, as this can be expensive only compute it if needed by output or visualization
                if (pRecording_params->outputAlignedFaces() || pRecording_params->outputHOG() || pRecording_params->outputAUs() || pVisualizer->vis_align || pVisualizer->vis_hog)
                {
                    pFace_analyser->PredictStaticAUsAndComputeFeatures(captured_image, vFace_models[model].detected_landmarks);
                    pFace_analyser->GetLatestAlignedFace(sim_warped_img);
                    pFace_analyser->GetLatestHOG(hog_descriptor, num_hog_rows, num_hog_cols);
                }
                */
                
                // Visualize the features
                pVisualizer->SetObservationFaceAlign(sim_warped_img);
                pVisualizer->SetObservationHOG(hog_descriptor, num_hog_rows, num_hog_cols);
                pVisualizer->SetObservationLandmarks(vFace_models[model].detected_landmarks, vFace_models[model].detection_certainty);
                pVisualizer->SetObservationPose(LandmarkDetector::GetPose(vFace_models[model], fx, fy, cx, cy), vFace_models[model].detection_certainty);
                pVisualizer->SetObservationGaze(gaze_direction0, gaze_direction1, LandmarkDetector::CalculateAllEyeLandmarks(vFace_models[model]), LandmarkDetector::Calculate3DEyeLandmarks(vFace_models[model], fx, fy, cx, cy), vFace_models[model].detection_certainty);
                pVisualizer->SetObservationActionUnits(pFace_analyser->GetCurrentAUsReg(), pFace_analyser->GetCurrentAUsClass());
                
                /*
                // Output features
                pOpen_face_rec->SetObservationHOG(vFace_models[model].detection_success, hog_descriptor, num_hog_rows, num_hog_cols, 31); // The number of channels in HOG is fixed at the moment, as using FHOG
                pOpen_face_rec->SetObservationVisualization(pVisualizer->GetVisImage());
                pOpen_face_rec->SetObservationActionUnits(pFace_analyser->GetCurrentAUsReg(), pFace_analyser->GetCurrentAUsClass());
                pOpen_face_rec->SetObservationLandmarks(vFace_models[model].detected_landmarks, vFace_models[model].GetShape(fx, fy, cx, cy),
                                                      vFace_models[model].params_global, vFace_models[model].params_local, vFace_models[model].detection_certainty, vFace_models[model].detection_success);
                pOpen_face_rec->SetObservationPose(pose_estimate);
                pOpen_face_rec->SetObservationGaze(gaze_direction0, gaze_direction1, gaze_angle, LandmarkDetector::CalculateAllEyeLandmarks(vFace_models[model]), LandmarkDetector::Calculate3DEyeLandmarks(vFace_models[model], fx, fy, cx, cy));
                pOpen_face_rec->SetObservationFaceAlign(sim_warped_img);
                pOpen_face_rec->SetObservationFaceID(model);
                pOpen_face_rec->SetObservationTimestamp(30); // bogus value
                pOpen_face_rec->SetObservationFrameNumber(30); // bogus value
                pOpen_face_rec->WriteObservation();
                */
            }
        }
    }
    
    // Update the frame count
    nFrameCount++;
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
    delete pVisualizer;
}

void ofxOpenFace::resetFaceModel() {
    for(size_t i=0; i < vFace_models.size(); ++i)
    {
        vFace_models[i].Reset();
        vActiveModels[i] = false;
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
                processImageMultipleFaces();
            } else {
                processImageSingleFace();
            }
            mutexImage.unlock();
            bHaveNewImage = false; // ready for a new image
        }
        
        int timeThisRunMs = ofGetElapsedTimeMillis() - timeBeforeMs;
        fTimePerRunMs = (fTimePerRunMs * fSmoothing) + (timeThisRunMs * (1.0f - fSmoothing)); // time in ms
        setFPS(1000.0f / fTimePerRunMs);
    }
    bHaveNewImage = false;
}

int ofxOpenFace::getFPS() {
    mutexFPS.lock();
    int nToReturn = (int)fFPS;
    mutexFPS.unlock();
    return nToReturn;
}

void ofxOpenFace::setFPS(float value) {
    mutexFPS.lock();
    fFPS = value;
    mutexFPS.unlock();
}

void ofxOpenFace::NonOverlapingDetections(const vector<LandmarkDetector::CLNF>& clnf_models, vector<cv::Rect_<double> >& face_detections)
{
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
