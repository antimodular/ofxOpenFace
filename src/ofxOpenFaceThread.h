// simple OF threading for OpenFace
namespace ofxOpenFace{
    class Thread : public ofThread{public:
#if OPENFACE_USE_MULTI
        MultiFace openFace;
#else
        SingleFace openFace;
#endif
        ofPixels pixels;
        bool bFrameNew = true;
        void threadedFunction(){
            lock();
            openFace.setup();
            unlock();
            while(isThreadRunning()){
                lock();
                if (bFrameNew){
                    openFace.setImage(pixels);
                }
                unlock();
            }
        }
    };
}
