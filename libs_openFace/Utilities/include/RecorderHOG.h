///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Tadas Baltrusaitis all rights reserved.
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

#ifndef __RECORDER_HOG_h_
#define __RECORDER_HOG_h_

// System includes
#include <vector>

// OpenCV includes
#include <opencv2/core/core.hpp>

#include <iostream>
#include <fstream>

namespace Utilities
{

	//===========================================================================
	/**
	A class for recording CSV file from OpenFace
	*/
	class RecorderHOG {

	public:

		// The constructor for the recorder, by default does not do anything
		RecorderHOG();
		
		// Adding observations to the recorder
		void SetObservationHOG(bool success, const cv::Mat_<double>& hog_descriptor, int num_cols, int num_rows, int num_channels);

		void Write();

		bool Open(std::string filename);

		void Close();

	private:

		// Blocking copy and move, as it doesn't make sense to read to write to the same file
		RecorderHOG & operator= (const RecorderHOG& other);
		RecorderHOG & operator= (const RecorderHOG&& other);
		RecorderHOG(const RecorderHOG&& other);
		RecorderHOG(const RecorderHOG& other);

		std::ofstream hog_file;

		// Internals for recording
		int num_cols;
		int num_rows;
		int num_channels;
		cv::Mat_<double> hog_descriptor;
		bool good_frame;

	};
}
#endif