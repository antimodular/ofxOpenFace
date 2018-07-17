namespace ofxOpenFace{
    // data structure for information on a face being tracked
    class trackedFace{public:
        int age = 0;
        int missingFrames = 0;
        int label = -1;
        ofVec2f lastPos;
        ofVec2f currPos;
    };

    // giving labels (IDs) to detected faces for tracking
    class Tracker{public:
        vector<trackedFace> tracked_faces;
        float movement_thresh = 200;
        float missing_thresh = 30;
        int label_distro = 0;
        bool isInit = false;
        
        void update(vector<faceData> &faces){
            if (!isInit){
                for (int i = 0; i < faces.size(); i++){
                    trackedFace tf;
                    tracked_faces.push_back(tf);
                }
                isInit = true;
            }
            for (int i = 0; i < tracked_faces.size(); i++){
                if (!faces[i].active){
                    tracked_faces[i].missingFrames ++;
                }else{
                    tracked_faces[i].missingFrames = 0;
                    tracked_faces[i].currPos = ofVec2f(faces[i].rect.x,faces[i].rect.y);
                    float d = tracked_faces[i].currPos.distance(tracked_faces[i].lastPos);
                    
                    if (d > movement_thresh || tracked_faces[i].label == -1){
                        ofLog()<<"NEW FACE: "<<ofToString(label_distro);
                        tracked_faces[i].label = label_distro++;
                    }
                    tracked_faces[i].lastPos = ofVec2f(tracked_faces[i].currPos.x,tracked_faces[i].currPos.y);
                }
                if (tracked_faces[i].missingFrames > missing_thresh){
                    tracked_faces[i].label = -1;
                    tracked_faces[i].currPos = ofVec2f(0,0);
                    tracked_faces[i].lastPos = ofVec2f(0,0);
                }
            }
        }
    };
}
