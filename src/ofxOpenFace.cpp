#include "ofxOpenFace.h"

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
    imgGrayScale.allocate(nImgWidth, nImgHeight, ofImageType::OF_IMAGE_COLOR);
    imgVisualized.allocate(nImgWidth, nImgHeight, ofImageType::OF_IMAGE_COLOR);
    
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
    
    // A utility for visualizing the results (show just the tracks)
    pVisualizer = new Utilities::Visualizer(true, false, false, false);
}

void ofxOpenFace::processImage(cv::Mat mat) {
    // Generate grayscale from img
    //ofxCv::toOf(mat, imgGrayScale);
    //imgGrayScale.setImageType(ofImageType::OF_IMAGE_GRAYSCALE);
    
    // Initialize some parameters. See https://github.com/TadasBaltrusaitis/OpenFace/wiki/API-calls
    float fx = 500.0f;
    float fy = 500.0f;
    float cx = (float)nImgWidth/2.0f;
    float cy = (float)nImgHeight/2.0f;
    
    // Reading the images
    cv::Mat captured_image = mat;
    //cv::Mat grayscale_image = ofxCv::toCv(imgGrayScale.getPixels());
    
    // The actual facial landmark detection / tracking
    //faceData.detected = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, *pFace_model, *pDet_parameters);
    faceData.detected = LandmarkDetector::DetectLandmarksInVideo(captured_image, *pFace_model, *pDet_parameters);
    
    // If tracking succeeded and we have an eye model, estimate gaze
    if (faceData.detected && pFace_model->eye_model)
    {
        GazeAnalysis::EstimateGaze(*pFace_model, faceData.gazeLeftEye, fx, fy, cx, cy, true);
        GazeAnalysis::EstimateGaze(*pFace_model, faceData.gazeRightEye , fx, fy, cx, cy, false);
    }
    faceData.certainty = pFace_model->detection_certainty;
    
    // Work out the pose of the head from the tracked model
    faceData.pose = LandmarkDetector::GetPose(*pFace_model, fx, fy, cx, cy);
    
    // Displaying the tracking visualizations
    mutexVisualizer.lock();
    pVisualizer->SetImage(captured_image, fx, fy, cx, cy);
    pVisualizer->SetObservationLandmarks(pFace_model->detected_landmarks, faceData.certainty, pFace_model->GetVisibilities());
    pVisualizer->SetObservationPose(faceData.pose, faceData.certainty);
    faceData.eyeLandmarks2D = LandmarkDetector::CalculateAllEyeLandmarks(*pFace_model);
    faceData.eyeLandmarks3D = LandmarkDetector::Calculate3DEyeLandmarks(*pFace_model, fx, fy, cx, cy);
    pVisualizer->SetObservationGaze(faceData.gazeLeftEye, faceData.gazeRightEye, faceData.eyeLandmarks2D, faceData.eyeLandmarks3D, faceData.certainty);
    mutexVisualizer.unlock();
}

void ofxOpenFace::setImage(cv::Mat mat) {
    mutexImage.lock();
    // Override the current "next image"
    matToProcess = mat;
    bHaveNewImage = true;
    mutexImage.unlock();
}

void ofxOpenFace::setImage(ofImage img) {
    img.setImageType(ofImageType::OF_IMAGE_GRAYSCALE);
    setImage(img.getPixels());
}

void ofxOpenFace::setImage(ofPixels pix) {
    mutexImage.lock();
    // Override the current "next image"
    matToProcess = ofxCv::toCv(pix);
    bHaveNewImage = true;
    mutexImage.unlock();
}

void ofxOpenFace::draw(int x, int y) {
    ofSetColor(ofColor::white);
    // Draw the visualization
    mutexVisualizer.lock();
    ofxCv::toOf(pVisualizer->GetVisImage(), imgVisualized);
    imgVisualized.draw(x, y);
    mutexVisualizer.unlock();
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

const OpenFaceData& ofxOpenFace::getFaceData() {
    return faceData;
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
            processImage(matToProcess);
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
