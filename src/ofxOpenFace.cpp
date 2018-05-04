#include "ofxOpenFace.h"

ofEvent<OpenFaceData> ofxOpenFace::eventDataReady = ofEvent<OpenFaceData>();

// Constructor
ofxOpenFace::ofxOpenFace(){
}

// Destructor
ofxOpenFace::~ofxOpenFace(){
    waitForThread(true);
}

void ofxOpenFace::setup(int nWidth, int nHeight) {
    nImgWidth = nWidth;
    nImgHeight = nHeight;
    
    // Set up OpenFace
    vector<string> arguments;
    arguments.push_back("-device");
    arguments.push_back("0");
    pDet_parameters = new LandmarkDetector::FaceModelParameters(arguments);
    
    // The modules that are being used for tracking
    string modelLocation = ofFilePath::getAbsolutePath("model/main_clnf_general.txt");
    pFace_model = new LandmarkDetector::CLNF(modelLocation);
    
    if (!pFace_model->eye_model) {
        ofLogError("No eye model found.");
    }
    
    // A utility for visualizing the results
    pVisualizer = new Utilities::Visualizer(true, false, false, false);
    bDoVisualizer = false;
}

void ofxOpenFace::processImage() {
    // Initialize some parameters. See https://github.com/TadasBaltrusaitis/OpenFace/wiki/API-calls
    float fx = 500.0f;
    float fy = 500.0f;
    float cx = (float)nImgWidth/2.0f;
    float cy = (float)nImgHeight/2.0f;
    
    // Reading the images
    cv::Mat captured_image = matToProcessColor;
    cv::Mat grayscale_image = matToProcessGrayScale;
    
    // The actual facial landmark detection / tracking
    OpenFaceData faceData;
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
    
    if (bDoVisualizer) {
        // Displaying the tracking visualizations
        pVisualizer->SetImage(captured_image, fx, fy, cx, cy);
        pVisualizer->SetObservationLandmarks(pFace_model->detected_landmarks, faceData.certainty, pFace_model->GetVisibilities());
        pVisualizer->SetObservationPose(faceData.pose, faceData.certainty);
        faceData.eyeLandmarks2D = LandmarkDetector::CalculateAllEyeLandmarks(*pFace_model);
        faceData.eyeLandmarks3D = LandmarkDetector::Calculate3DEyeLandmarks(*pFace_model, fx, fy, cx, cy);
        pVisualizer->SetObservationGaze(faceData.gazeLeftEye, faceData.gazeRightEye, faceData.eyeLandmarks2D, faceData.eyeLandmarks3D, faceData.certainty);
    }
    
    ofNotifyEvent(eventDataReady, faceData);
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
    delete pFace_model;
    delete pDet_parameters;
    delete pVisualizer;
}

void ofxOpenFace::resetFaceModel() {
    pFace_model->Reset();    
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
            processImage();
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
