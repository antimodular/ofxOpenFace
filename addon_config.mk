meta:
	ADDON_NAME = ofxOpenFace
	ADDON_DESCRIPTION = openframeworks wrapper for OpenFace
	ADDON_AUTHOR = Antimodular Research
	ADDON_TAGS = "openface"
	ADDON_URL = https://github.com/antimodular/ofxOpenFace

common:
	# dependencies with other addons, a list of them separated by spaces 
	# or use += in several lines
	ADDON_DEPENDENCIES = ofxCv
	
	# include search paths, this will be usually parsed from the file system
	# but if the addon or addon libraries need special search paths they can be
	# specified here separated by spaces or one per line using +=
	ADDON_INCLUDES = libs/FaceAnalyser/include
	ADDON_INCLUDES += libs/GazeAnalyser/include
    	ADDON_INCLUDES += libs/LandmarkDetector/include
    	ADDON_INCLUDES += libs/Utilities/include
    	ADDON_INCLUDES += libs/tbb/include
    	ADDON_INCLUDES += libs/opencv3.4.1/include
	ADDON_INCLUDES += libs_others/dlib/include/dlib/all/source.cpp
	ADDON_INCLUDES += libs_others/dlib/include
	ADDON_INCLUDES += src

	# any special flag that should be passed to the compiler when using this
	# addon
	ADDON_CFLAGS = 
	
	# any special flag that should be passed to the linker when using this
	# addon, also used for system libraries with -lname
	ADDON_LDFLAGS = 
	
	# linux only, any library that should be included in the project using
	# pkg-config
	ADDON_PKG_CONFIG_LIBRARIES =
	
	# osx/iOS only, any framework that should be included in the project
	ADDON_FRAMEWORKS =
	
	# source files, these will be usually parsed from the file system looking
	# in the src folders in libs and the root of the addon. if your addon needs
	# to include files in different places or a different set of files per platform
	# they can be specified here

	# some addons need resources to be copied to the bin/data folder of the project
	# specify here any files that need to be copied, you can use wildcards like * and ?
	ADDON_DATA = 
	
	# when parsing the file system looking for libraries exclude this for all or
	# a specific platform
	ADDON_LIBS_EXCLUDE =
	
linux64:
	# binary libraries, these will be usually parsed from the file system but some 
	# libraries need to passed to the linker in a specific order 
	#nothing yet
	
linux:
	#nothing yet

vs:
    	#nothing yet

linuxarmv6l:
    	#nothing yet
linuxarmv7l:
	#nothing yet
	
android/armeabi:	
	#nothing yet
	
android/armeabi-v7a:	
	#nothing yet

osx:
	# Required libraries
	ADDON_LIBS = libs/FaceAnalyser/lib/osx/libFaceAnalyser.a
	ADDON_LIBS += libs/GazeAnalyser/lib/osx/libGazeAnalyser.a
	ADDON_LIBS += libs/LandmarkDetector/lib/osx/libLandmarkDetector.a
	ADDON_LIBS += libs/Utilities/lib/osx/libUtilities.a
	ADDON_LIBS += libs/opencv3.4.1/lib/osx/libopencv_calib3d.dylib
	ADDON_LIBS += libs/opencv3.4.1/lib/osx/libopencv_core.dylib
	ADDON_LIBS += libs/opencv3.4.1/lib/osx/libopencv_dnn.dylib
	ADDON_LIBS += libs/opencv3.4.1/lib/osx/libopencv_features2d.dylib
	ADDON_LIBS += libs/opencv3.4.1/lib/osx/libopencv_flann.dylib
	ADDON_LIBS += libs/opencv3.4.1/lib/osx/libopencv_highgui.dylib
	ADDON_LIBS += libs/opencv3.4.1/lib/osx/libopencv_imgcodecs.dylib
	ADDON_LIBS += libs/opencv3.4.1/lib/osx/libopencv_imgproc.dylib
	ADDON_LIBS += libs/opencv3.4.1/lib/osx/libopencv_ml.dylib
	ADDON_LIBS += libs/opencv3.4.1/lib/osx/libopencv_objdetect.dylib
	ADDON_LIBS += libs/opencv3.4.1/lib/osx/libopencv_photo.dylib
	ADDON_LIBS += libs/opencv3.4.1/lib/osx/libopencv_shape.dylib
	ADDON_LIBS += libs/opencv3.4.1/lib/osx/libopencv_stitching.dylib
	ADDON_LIBS += libs/opencv3.4.1/lib/osx/libopencv_superres.dylib
	ADDON_LIBS += libs/opencv3.4.1/lib/osx/libopencv_video.dylib
	ADDON_LIBS += libs/opencv3.4.1/lib/osx/libopencv_videoio.dylib
	ADDON_LIBS += libs/opencv3.4.1/lib/osx/libopencv_videostab.dylib
	ADDON_LIBS += libs_others/dlib/lib/osx/libdlib.a
    	ADDON_LIBS += libs/tbb/lib/osx/libtbb.dylib

ios:
	#nothing yet
