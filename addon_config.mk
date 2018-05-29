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
	ADDON_INCLUDES = libs_openFace/FaceAnalyser/include
	ADDON_INCLUDES += libs_openFace/GazeAnalyser/include
    	ADDON_INCLUDES += libs_openFace/LandmarkDetector/include
    	ADDON_INCLUDES += libs_openFace/Utilities/include
	ADDON_INCLUDES += libs_others/dlib/include/dlib/all/source.cpp
	ADDON_INCLUDES += libs_others/dlib/include
    	ADDON_INCLUDES += /usr/local/opt/openblas/include/
    	ADDON_INCLUDES += /usr/local/include
	ADDON_INCLUDES += src

	# any special flag that should be passed to the compiler when using this
	# addon
	ADDON_CFLAGS = 
	
	# any special flag that should be passed to the linker when using this
	# addon, also used for system libraries with -lname
	#ADDON_LDFLAGS = -rpath ../../../../addons/ofxOpenFace/libs_others/opencv3.4.1/lib/osx
	
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
	ADDON_LIBS = libs_openFace/FaceAnalyser/lib/osx/libFaceAnalyser.a
	ADDON_LIBS += libs_openFace/GazeAnalyser/lib/osx/libGazeAnalyser.a
	ADDON_LIBS += libs_openFace/LandmarkDetector/lib/osx/libLandmarkDetector.a
	ADDON_LIBS += libs_openFace/Utilities/lib/osx/libUtilities.a
	ADDON_LIBS += libs_others/dlib/lib/osx/libdlib.a
	ADDON_LIBS += /usr/local/opt/openblas/lib/libopenblas.a
	ADDON_LIBS += /usr/local/lib/libopencv_calib3d.dylib
	ADDON_LIBS += /usr/local/lib/libopencv_core.dylib
	ADDON_LIBS += /usr/local/lib/libopencv_dnn.dylib
	ADDON_LIBS += /usr/local/lib/libopencv_features2d.dylib
	ADDON_LIBS += /usr/local/lib/libopencv_flann.dylib
	ADDON_LIBS += /usr/local/lib/libopencv_highgui.dylib
	ADDON_LIBS += /usr/local/lib/libopencv_imgcodecs.dylib
	ADDON_LIBS += /usr/local/lib/libopencv_imgproc.dylib
	ADDON_LIBS += /usr/local/lib/libopencv_ml.dylib
	ADDON_LIBS += /usr/local/lib/libopencv_objdetect.dylib
	ADDON_LIBS += /usr/local/lib/libopencv_photo.dylib
	ADDON_LIBS += /usr/local/lib/libopencv_shape.dylib
	ADDON_LIBS += /usr/local/lib/libopencv_stitching.dylib
	ADDON_LIBS += /usr/local/lib/libopencv_superres.dylib
	ADDON_LIBS += /usr/local/lib/libopencv_video.dylib
	ADDON_LIBS += /usr/local/lib/libopencv_videoio.dylib
	ADDON_LIBS += /usr/local/lib/libopencv_videostab.dylib
    	ADDON_LIBS += /usr/local/lib/libtbb.dylib

	#ADDON_AFTER = echo "Not implemented yet"
ios:
	#nothing yet
