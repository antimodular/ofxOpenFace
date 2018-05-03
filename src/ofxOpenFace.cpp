#include "ofxOpenFace.h"

void ofxOpenFace::setup(int nWidth, int nHeight) {
    nImgWidth = nWidth;
    nImgHeight = nHeight;
    imgGrayScale.allocate(nImgWidth, nImgHeight, ofImageType::OF_IMAGE_GRAYSCALE);
    
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

void ofxOpenFace::update(cv::Mat &mat) {
    // Generate grayscale from img
    ofxCv::toOf(mat, imgGrayScale);
    imgGrayScale.setImageType(ofImageType::OF_IMAGE_GRAYSCALE);
    
    // Initialize some parameters. See https://github.com/TadasBaltrusaitis/OpenFace/wiki/API-calls
    float fx = 500.0f;
    float fy = 500.0f;
    float cx = (float)nImgWidth/2.0f;
    float cy = (float)nImgHeight/2.0f;
    
    // Reading the images
    cv::Mat captured_image = mat;
    cv::Mat grayscale_image = ofxCv::toCv(imgGrayScale.getPixels());
    
    // The actual facial landmark detection / tracking
    faceData.detected = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, *pFace_model, *pDet_parameters);
    
    // Gaze tracking, absolute gaze direction
    cv::Point3f gazeDirection0(0, 0, -1);
    cv::Point3f gazeDirection1(0, 0, -1);
    
    // If tracking succeeded and we have an eye model, estimate gaze
    if (faceData.detected && pFace_model->eye_model)
    {
        GazeAnalysis::EstimateGaze(*pFace_model, gazeDirection0, fx, fy, cx, cy, true);
        GazeAnalysis::EstimateGaze(*pFace_model, gazeDirection1, fx, fy, cx, cy, false);
        faceData.gazeLeftEye = gazeDirection0;
        faceData.gazeRightEye = gazeDirection1;
    }
    faceData.certainty = pFace_model->detection_certainty;
    
    // Work out the pose of the head from the tracked model
    faceData.pose = LandmarkDetector::GetPose(*pFace_model, fx, fy, cx, cy);
    
    // Displaying the tracking visualizations
    pVisualizer->SetImage(captured_image, fx, fy, cx, cy);
    pVisualizer->SetObservationLandmarks(pFace_model->detected_landmarks, faceData.certainty, pFace_model->GetVisibilities());
    pVisualizer->SetObservationPose(faceData.pose, faceData.certainty);
    faceData.eyeLandmarks2D = LandmarkDetector::CalculateAllEyeLandmarks(*pFace_model);
    faceData.eyeLandmarks3D = LandmarkDetector::Calculate3DEyeLandmarks(*pFace_model, fx, fy, cx, cy);
    pVisualizer->SetObservationGaze(faceData.gazeLeftEye, faceData.gazeRightEye, faceData.eyeLandmarks2D, faceData.eyeLandmarks3D, faceData.certainty);
}

void ofxOpenFace::update(ofImage &img) {
    update(img.getPixels());
}

void ofxOpenFace::update(ofPixels& pix) {
    cv::Mat captured_image = ofxCv::toCv(pix);
    update(captured_image);
}

void ofxOpenFace::draw() {
    ofSetColor(ofColor::white);
    // Draw the visualization
    ofImage imgVisualized;
    ofxCv::toOf(pVisualizer->GetVisImage(), imgVisualized);
    imgVisualized.draw(20, 20);
}

void ofxOpenFace::exit() {
    // Clear memory
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
