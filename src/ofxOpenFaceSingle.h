//single face detection
//result stored in .face
namespace ofxOpenFace{
    class SingleFace : public BaseFace{public:

        faceData face;
        
        void setup(){
            BaseFace::setup();
            
            // The modules that are being used for tracking
            face_model = LandmarkDetector::CLNF(det_parameters.model_location);
            if (!face_model.loaded_successfully)
            {
                cout << "ERROR: Could not load the landmark detector" << endl;
                return 1;
            }
            
            if (!face_model.eye_model)
            {
                cout << "WARNING: no eye model found" << endl;
            }
            
            // Tracking FPS for visualization
            fps_tracker.AddFrame();
            
        }

        void setImage(cv::Mat rgb_image){
            
            cv::Mat_<uchar> grayscale_image = sequence_reader.GetGrayFrame(rgb_image);
            
            // The actual facial landmark detection / tracking
            bool detection_success = LandmarkDetector::DetectLandmarksInVideo(rgb_image, face_model, det_parameters, grayscale_image);
            
            // Gaze tracking, absolute gaze direction
            cv::Point3f gazeDirection0(0, 0, -1);
            cv::Point3f gazeDirection1(0, 0, -1);
            
            // If tracking succeeded and we have an eye model, estimate gaze
            if (detection_success && face_model.eye_model)
            {
                GazeAnalysis::EstimateGaze(face_model, gazeDirection0, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, true);
                GazeAnalysis::EstimateGaze(face_model, gazeDirection1, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, false);
            }
            
            // Work out the pose of the head from the tracked model
            cv::Vec6d pose_estimate = LandmarkDetector::GetPose(face_model, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);
            
            // Keeping track of FPS
            fps_tracker.AddFrame();
            
            // Displaying the tracking visualizations
            visualizer.SetImage(rgb_image, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);
            visualizer.SetObservationLandmarks(face_model.detected_landmarks, face_model.detection_certainty, face_model.GetVisibilities());
            visualizer.SetObservationPose(pose_estimate, face_model.detection_certainty);
            visualizer.SetObservationGaze(gazeDirection0, gazeDirection1, LandmarkDetector::CalculateAllEyeLandmarks(face_model), LandmarkDetector::Calculate3DEyeLandmarks(face_model, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy), face_model.detection_certainty);
            
            fps = fps_tracker.GetFPS();
            visualizer.SetFps(fps);
            // detect key presses (due to pecularities of OpenCV, you can get it when displaying images)
            // char character_press = visualizer.ShowObservation();
            
            face.pose = pose_estimate;
            face.landmarks = &(face_model.detected_landmarks);
            canvas = visualizer.captured_image;
            face.gaze0 = gazeDirection0;
            face.gaze1 = gazeDirection1;
            face.rect = face_model.GetBoundingBox();
            face.certainty = face_model.detection_certainty;
            face.active = true;
            
        }
        void setImage(ofPixels pixels){
            mat_rgb = ofxCv::toCv(pixels);
            ofxCv::convertColor(mat_rgb,mat_bgr,CV_BGR2RGB);
            setImage(mat_bgr);
        }
        void setImage(ofImage img){
            ofpix = img.getPixels();
            setImage(ofpix);
        }
        void reset(){
            face_model.Reset();
        }
    };
}
