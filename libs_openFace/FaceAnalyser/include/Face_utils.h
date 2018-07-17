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
//       OpenFace 2.0: Facial Behavior Analysis Toolkit
//       Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018  
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltrušaitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.    
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
///////////////////////////////////////////////////////////////////////////////

#ifndef __FACE_UTILS_h_
#define __FACE_UTILS_h_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "PDM.h"
#include "PAW.h"

namespace FaceAnalysis
{
	//===========================================================================	
	// Defining a set of useful utility functions to be used within FaceAnalyser

	// Aligning a face to a common reference frame
	void AlignFace(cv::Mat& aligned_face, const cv::Mat& frame, const cv::Mat_<float>& detected_landmarks, cv::Vec6f params_global, const LandmarkDetector::PDM& pdm, bool rigid = true, double scale = 0.7, int width = 96, int height = 96);
	void AlignFaceMask(cv::Mat& aligned_face, const cv::Mat& frame, const cv::Mat_<float>& detected_landmarks, cv::Vec6f params_global, const LandmarkDetector::PDM& pdm, const cv::Mat_<int>& triangulation, bool rigid = true, double scale = 0.7, int width = 96, int height = 96);

	void Extract_FHOG_descriptor(cv::Mat_<double>& descriptor, const cv::Mat& image, int& num_rows, int& num_cols, int cell_size = 8);

	// The following two methods go hand in hand
	void ExtractSummaryStatistics(const cv::Mat_<double>& descriptors, cv::Mat_<double>& sum_stats, bool mean, bool stdev, bool max_min);
	void AddDescriptor(cv::Mat_<double>& descriptors, cv::Mat_<double> new_descriptor, int curr_frame, int num_frames_to_keep = 120);

	//===========================================================================
	// Point set and landmark manipulation functions
	//===========================================================================
	// Using Kabsch's algorithm for aligning shapes
	//This assumes that align_from and align_to are already mean normalised
	cv::Matx22f AlignShapesKabsch2D(const cv::Mat_<float>& align_from, const cv::Mat_<float>& align_to);

	//=============================================================================
	// Basically Kabsch's algorithm but also allows the collection of points to be different in scale from each other
	cv::Matx22f AlignShapesWithScale(cv::Mat_<float>& src, cv::Mat_<float> dst);
	
	//============================================================================
	// Matrix reading functionality
	//============================================================================

	// Reading a matrix written in a binary format
	void ReadMatBin(std::ifstream& stream, cv::Mat &output_mat);

	// Reading in a matrix from a stream
	void ReadMat(std::ifstream& stream, cv::Mat& output_matrix);

	// Skipping comments (lines starting with # symbol)
	void SkipComments(std::ifstream& stream);

}
#endif
