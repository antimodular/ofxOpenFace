# ofxOpenFace
Toolkit capable of facial landmark detection, head pose estimation, facial action unit recognition, and eye-gaze estimation.

## Description
This addon for [openFrameworks](http://openframeworks.cc) implements the face detection and face landmark detection algorithm from [openFace](https://github.com/TadasBaltrusaitis/OpenFace). 
We hope the addon is similar to [ofxFaceTracker](https://github.com/kylemcdonald/ofxFaceTracker) and [ofxFaceTracker2](https://github.com/HalfdanJ/ofxFaceTracker2).


The addon is able to:
- receive any ofImage, ofPixel or cv::Mat
- find all faces
- collect all landmarks and provide their 2D and 3D points
- label each face with an id
- provide the age of each label
- for tracking the faces over consecutive frame we use Kyle’s [ofxCv tracker](https://github.com/kylemcdonald/ofxCv/blob/master/libs/ofxCv/include/ofxCv/Tracker.h)

Check the wiki for more instruction: https://github.com/antimodular/ofxOpenFace/wiki
