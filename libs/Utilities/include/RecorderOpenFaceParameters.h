///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Tadas Baltrusaitis, all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt
//
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace: an open source facial behavior analysis toolkit
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency
//       in IEEE Winter Conference on Applications of Computer Vision, 2016  
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-speci?c normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency. 
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
///////////////////////////////////////////////////////////////////////////////

//  Parameters of the Face analyser
#ifndef __RECORDER_OPENFACE_PARAM_H
#define __RECORDER_OPENFACE_PARAM_H

#include <vector>
#include <opencv2/core/core.hpp>

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

using namespace std;

namespace Utilities
{

	class RecorderOpenFaceParameters
	{

	public:

		// Constructors
		RecorderOpenFaceParameters(std::vector<std::string> &arguments, bool sequence, bool is_from_webcam, float fx = -1, float fy = -1, float cx = -1, float cy = -1, double fps_vid_out = 30);
		RecorderOpenFaceParameters(bool sequence, bool is_from_webcam, bool output_2D_landmarks, bool output_3D_landmarks,
			bool output_model_params, bool output_pose, bool output_AUs, bool output_gaze, bool output_hog, bool output_tracked,
			bool output_aligned_faces, float fx = -1, float fy = -1, float cx = -1, float cy = -1, double fps_vid_out = 30);

		bool isSequence() const { return is_sequence; }
		bool isFromWebcam() const { return is_from_webcam; }
		bool output2DLandmarks() const { return output_2D_landmarks; }
		bool output3DLandmarks() const { return output_3D_landmarks; }
		bool outputPDMParams() const { return output_model_params; }
		bool outputPose() const { return output_pose; }
		bool outputAUs() const { return output_AUs; }
		bool outputGaze() const { return output_gaze; }
		bool outputHOG() const { return output_hog; }
		bool outputTracked() const { return output_tracked; }
		bool outputAlignedFaces() const { return output_aligned_faces; }
		std::string outputCodec() const { return output_codec; }
		double outputFps() const { return fps_vid_out; }

		float getFx() const { return fx; }
		float getFy() const { return fy; }
		float getCx() const { return cx; }
		float getCy() const { return cy; }

		void setOutputAUs(bool output_AUs) { this->output_AUs = output_AUs; }
		void setOutputGaze(bool output_gaze) { this->output_gaze = output_gaze; }

	private:
		
		// If we are recording results from a sequence each row refers to a frame, if we are recording an image each row is a face
		bool is_sequence;
		// If the data is coming from a webcam
		bool is_from_webcam;

		// Keep track of what we are recording
		bool output_2D_landmarks;
		bool output_3D_landmarks;
		bool output_model_params;
		bool output_pose;
		bool output_AUs;
		bool output_gaze;
		bool output_hog;
		bool output_tracked;
		bool output_aligned_faces;
		
		// Some video recording parameters
		std::string output_codec;
		double fps_vid_out;

		// Camera parameters for recording in the meta file;
		float fx, fy, cx, cy;

	};

}

#endif // ____RECORDER_OPENFACE_PARAM_H
