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

#ifndef __VISUALIZATION_UTILS_h_
#define __VISUALIZATION_UTILS_h_

#include <opencv2/core/core.hpp>

#include <vector>
#include <queue>

namespace Utilities
{

	// TODO draw AU results

	// Drawing a bounding box around the face in an image
	void DrawBox(cv::Mat image, cv::Vec6d pose, cv::Scalar color, int thickness, float fx, float fy, float cx, float cy);
	void DrawBox(const std::vector<std::pair<cv::Point2d, cv::Point2d>>& lines, cv::Mat image, cv::Scalar color, int thickness);

	// Computing a bounding box to be drawn
	std::vector<std::pair<cv::Point2d, cv::Point2d>> CalculateBox(cv::Vec6d pose, float fx, float fy, float cx, float cy);

    void Visualise_FHOG(const cv::Mat_<double>& descriptor, int num_rows, int num_cols, cv::Mat& visualisation);

	class FpsTracker
	{
	public:
		
		double history_length;

		void AddFrame();

		double GetFPS();

		FpsTracker();

	private:
		std::queue<double> frame_times;

		void DiscardOldFrames();

	};

}
#endif