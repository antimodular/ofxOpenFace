// multiple face detection
// result stored in .faces
namespace ofxOpenFace{
    class MultiFace: public BaseFace{
    public:
        
        vector<faceData> faces;
        vector<LandmarkDetector::CLNF> face_models;
        vector<bool> active_models;
        int num_faces_max = 4;
        int frame_count;
        
        void NonOverlapingDetections(const vector<LandmarkDetector::CLNF>& clnf_models, vector<cv::Rect_<float> >& face_detections)
        {
            
            // Go over the model and eliminate detections that are not informative (there already is a tracker there)
            for (size_t model = 0; model < clnf_models.size(); ++model)
            {
                
                // See if the detections intersect
                cv::Rect_<float> model_rect = clnf_models[model].GetBoundingBox();
                
                for (int detection = face_detections.size() - 1; detection >= 0; --detection)
                {
                    double intersection_area = (model_rect & face_detections[detection]).area();
                    double union_area = model_rect.area() + face_detections[detection].area() - 2 * intersection_area;
                    
                    // If the model is already tracking what we're detecting ignore the detection, this is determined by amount of overlap
                    if (intersection_area / union_area > 0.5)
                    {
                        face_detections.erase(face_detections.begin() + detection);
                    }
                }
            }
        }
        
        void setup(){
            
            BaseFace::setup();
            
            det_parameters.model_location = model_location+"/model/main_ceclm_general.txt";
            
            det_parameters.reinit_video_every = -1;
            det_parameters.curr_face_detector = LandmarkDetector::FaceModelParameters::MTCNN_DETECTOR;
            
            det_parameters.haar_face_detector_location = model_location+"/classifiers/haarcascade_frontalface_alt.xml";
            det_parameters.mtcnn_face_detector_location = model_location+"/model/mtcnn_detector/MTCNN_detector.txt";
            
            face_model = LandmarkDetector::CLNF(det_parameters.model_location);
            if (!face_model.loaded_successfully)
            {
                cout << "ERROR: Could not load the landmark detector" << endl;
                return 1;
            }
            
            face_model.face_detector_HAAR.load(det_parameters.haar_face_detector_location);
            face_model.haar_face_detector_location = det_parameters.haar_face_detector_location;
            face_model.face_detector_MTCNN.Read(det_parameters.mtcnn_face_detector_location);
            face_model.mtcnn_face_detector_location = det_parameters.mtcnn_face_detector_location;
            
            if (det_parameters.curr_face_detector == LandmarkDetector::FaceModelParameters::MTCNN_DETECTOR && face_model.face_detector_MTCNN.empty())
            {
                cout << "INFO: defaulting to HOG-SVM face detector" << endl;
                det_parameters.curr_face_detector = LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR;
            }
            
            face_models.reserve(num_faces_max);
            //faces.reserve(num_faces_max);
            face_models.push_back(face_model);
            active_models.push_back(false);
            for (int i = 1; i < num_faces_max; ++i)
            {
                face_models.push_back(face_model);
                active_models.push_back(false);
                faceData fd;
                faces.push_back(fd);
            }
            frame_count = 0;
            fps_tracker.AddFrame();
            
        }
        void setImage(cv::Mat rgb_image){
            
            cv::Mat_<uchar> grayscale_image = sequence_reader.GetGrayFrame(rgb_image);
            vector<cv::Rect_<float> > face_detections;
            bool all_models_active = true;
            for (unsigned int model = 0; model < face_models.size(); ++model)
            {
                if (!active_models[model])
                {
                    all_models_active = false;
                }
            }
            // Get the detections (every 8th frame and when there are free models available for tracking)
            if (frame_count % 8 == 0 && !all_models_active)
            {
                if (det_parameters.curr_face_detector == LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR)
                {
                    vector<float> confidences;
                    LandmarkDetector::DetectFacesHOG(face_detections, grayscale_image, face_models[0].face_detector_HOG, confidences);
                }
                else if (det_parameters.curr_face_detector == LandmarkDetector::FaceModelParameters::HAAR_DETECTOR)
                {
                    LandmarkDetector::DetectFaces(face_detections, grayscale_image, face_models[0].face_detector_HAAR);
                }
                else
                {
                    vector<float> confidences;
                    LandmarkDetector::DetectFacesMTCNN(face_detections, rgb_image, face_models[0].face_detector_MTCNN, confidences);
                }
                
            }
            // Keep only non overlapping detections (also convert to a concurrent vector
            
            NonOverlapingDetections(face_models, face_detections);
            
            vector<tbb::atomic<bool> > face_detections_used(face_detections.size());
            
            
            // Go through every model and update the tracking
            //tbb::parallel_for(0, (int)face_models.size(), [&](int model) {
            for (unsigned int model = 0; model < face_models.size(); ++model)
            {
                
                bool detection_success = false;
                
                // If the current model has failed more than 4 times in a row, remove it
                if (face_models[model].failures_in_a_row > 4)
                {
                    active_models[model] = false;
                    face_models[model].Reset();
                }
                
                // If the model is inactive reactivate it with new detections
                if (!active_models[model])
                {
                    
                    for (size_t detection_ind = 0; detection_ind < face_detections.size(); ++detection_ind)
                    {
                        // if it was not taken by another tracker take it (if it is false swap it to true and enter detection, this makes it parallel safe)
                        if (face_detections_used[detection_ind].compare_and_swap(true, false) == false)
                        {
                            
                            // Reinitialise the model
                            face_models[model].Reset();
                            
                            // This ensures that a wider window is used for the initial landmark localisation
                            face_models[model].detection_success = false;
                            detection_success = LandmarkDetector::DetectLandmarksInVideo(rgb_image, face_detections[detection_ind], face_models[model], det_parameters, grayscale_image);
                            
                            // This activates the model
                            active_models[model] = true;
                            
                            // break out of the loop as the tracker has been reinitialised
                            break;
                        }
                        
                    }
                }
                else
                {
                    // The actual facial landmark detection / tracking
                    detection_success = LandmarkDetector::DetectLandmarksInVideo(rgb_image, face_models[model], det_parameters, grayscale_image);
                }
            }
            //});
            
            // Keeping track of FPS
            fps_tracker.AddFrame();
            
            visualizer.SetImage(rgb_image, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);
            
            // Go through every model and detect eye gaze, record results and visualise the results
            for (size_t model = 0; model < face_models.size(); ++model)
            {
                // Visualising and recording the results
                if (active_models[model])
                {
                    
                    // Estimate head pose and eye gaze
                    cv::Vec6d pose_estimate = LandmarkDetector::GetPose(face_models[model], sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);
                    
                    cv::Point3f gaze_direction0(0, 0, 0); cv::Point3f gaze_direction1(0, 0, 0); cv::Vec2d gaze_angle(0, 0);
                    
                    // Detect eye gazes
                    if (face_models[model].detection_success && face_model.eye_model)
                    {
                        GazeAnalysis::EstimateGaze(face_models[model], gaze_direction0, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, true);
                        GazeAnalysis::EstimateGaze(face_models[model], gaze_direction1, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, false);
                        gaze_angle = GazeAnalysis::GetGazeAngle(gaze_direction0, gaze_direction1);
                    }
                    
                    // Visualize the features
                    visualizer.SetObservationLandmarks(face_models[model].detected_landmarks, face_models[model].detection_certainty);
                    visualizer.SetObservationPose(LandmarkDetector::GetPose(face_models[model], sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy), face_models[model].detection_certainty);
                    visualizer.SetObservationGaze(gaze_direction0, gaze_direction1, LandmarkDetector::CalculateAllEyeLandmarks(face_models[model]), LandmarkDetector::Calculate3DEyeLandmarks(face_models[model], sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy), face_models[model].detection_certainty);
                    
                    faces[model].pose = pose_estimate;
                    faces[model].landmarks = &(face_models[model].detected_landmarks);
                    faces[model].gaze0 = gaze_direction0;
                    faces[model].gaze1 = gaze_direction1;
                    faces[model].rect = face_models[model].GetBoundingBox();
                    faces[model].certainty = face_models[model].detection_certainty;
                    faces[model].active = true;
                }else{
                    faces[model].active = false;
                }
            }
            
            fps = fps_tracker.GetFPS();
            visualizer.SetFps(fps);
            frame_count ++;
            
            canvas = visualizer.captured_image;
            
            
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
            for (size_t i = 0; i < face_models.size(); ++i)
            {
                face_models[i].Reset();
                active_models[i] = false;
            }
        }
        
        
    };
}
