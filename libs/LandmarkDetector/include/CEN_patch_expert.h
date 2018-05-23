///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
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


#ifndef __CEN_PATCH_EXPERT_h_
#define __CEN_PATCH_EXPERT_h_

// system includes
#include <vector>

// OpenCV includes
#include <opencv2/core/core.hpp>

namespace LandmarkDetector
{
	//===========================================================================
	/**
	The classes describing the CEN patch experts
	*/

	class CEN_patch_expert {
	public:

		// Width and height of the patch expert support area
		int width_support;
		int height_support;

		// Neural weights
		std::vector<cv::Mat_<float>> biases;

		// Neural weights
		std::vector<cv::Mat_<float>> weights;

		std::vector<int> activation_function;
		
		// Confidence of the current patch expert (used for NU_RLMS optimisation)
		double  confidence;

		CEN_patch_expert() { ; }

		// A copy constructor
		CEN_patch_expert(const CEN_patch_expert& other);

		// Reading in the patch expert
		void Read(std::ifstream &stream);

		// The actual response computation from intensity image
		void Response(const cv::Mat_<float> &area_of_interest, cv::Mat_<float> &response);

		// Faster version of the response that only considers a subset of the area_of_interest
		void ResponseSparse(const cv::Mat_<float> &area_of_interest, cv::Mat_<float> &response, cv::Mat_<float>& mapMatrix, cv::Mat_<float>& im2col_prealloc);

		// To save memory use a mirrored version of the expert instead of storing the weights
		void ResponseSparse_mirror(const cv::Mat_<float> &area_of_interest, cv::Mat_<float> &response, cv::Mat_<float>& mapMatrix, cv::Mat_<float>& im2col_prealloc);

		// For frontal faces can apply mirrored and non-mirrored experts at the same time
		void ResponseSparse_mirror_joint(const cv::Mat_<float> &area_of_interest_left, const cv::Mat_<float> &area_of_interest_right, cv::Mat_<float> &response_left, cv::Mat_<float> &response_right, cv::Mat_<float>& mapMatrix, cv::Mat_<float>& im2col_prealloc_left, cv::Mat_<float>& im2col_prealloc_right);

	};

	void interpolationMatrix(cv::Mat_<float>& mapMatrix, int response_height, int response_width, int input_width, int input_height);

}
#endif
