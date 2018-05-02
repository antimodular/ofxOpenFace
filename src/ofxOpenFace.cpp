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

void ofxOpenFace::update(ofImage &img) {
    imgGrayScale.clone(img);
    imgGrayScale.setImageType(ofImageType::OF_IMAGE_GRAYSCALE);
    
    // Initialize some parameters. See https://github.com/TadasBaltrusaitis/OpenFace/wiki/API-calls
    float fx = 500.0f;
    float fy = 500.0f;
    float cx = (float)nImgWidth/2.0f;
    float cy = (float)nImgHeight/2.0f;
    
    // Reading the images
    cv::Mat captured_image = ofxCv::toCv(img.getPixels());
    cv::Mat grayscale_image = ofxCv::toCv(imgGrayScale.getPixels());
    
    // The actual facial landmark detection / tracking
    bool detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, *pFace_model, *pDet_parameters);
    
    // Gaze tracking, absolute gaze direction
    cv::Point3f gazeDirection0(0, 0, -1);
    cv::Point3f gazeDirection1(0, 0, -1);
    
    // If tracking succeeded and we have an eye model, estimate gaze
    if (detection_success && pFace_model->eye_model)
    {
        GazeAnalysis::EstimateGaze(*pFace_model, gazeDirection0, fx, fy, cx, cy, true);
        GazeAnalysis::EstimateGaze(*pFace_model, gazeDirection1, fx, fy, cx, cy, false);
    }
    
    // Work out the pose of the head from the tracked model
    cv::Vec6d pose_estimate = LandmarkDetector::GetPose(*pFace_model, fx, fy, cx, cy);
    
    // Displaying the tracking visualizations
    pVisualizer->SetImage(captured_image, fx, fy, cx, cy);
    pVisualizer->SetObservationLandmarks(pFace_model->detected_landmarks, pFace_model->detection_certainty, pFace_model->GetVisibilities());
    pVisualizer->SetObservationPose(pose_estimate, pFace_model->detection_certainty);
    pVisualizer->SetObservationGaze(gazeDirection0, gazeDirection1, LandmarkDetector::CalculateAllEyeLandmarks(*pFace_model), LandmarkDetector::Calculate3DEyeLandmarks(*pFace_model, fx, fy, cx, cy), pFace_model->detection_certainty);
}


void ofxOpenFace::draw() {
    ofSetColor(ofColor::white);
    // Draw the visualization
    cv::Mat matVisualized = pVisualizer->GetVisImage();
    ofImage imgVisualized;
    ofxCv::toOf(matVisualized, imgVisualized);
    imgVisualized.draw(20, 20);
}

void ofxOpenFace::resetFaceModel() {
    pFace_model->Reset();    
}
