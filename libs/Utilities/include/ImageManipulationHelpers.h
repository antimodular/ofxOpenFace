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

#ifndef __IMAGE_MANIPULATION_HELPERS_h_
#define __IMAGE_MANIPULATION_HELPERS_h_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

namespace Utilities
{
	//===========================================================================
	// Converting between color spaces and bit depths
	//===========================================================================

	static void ConvertToGrayscale_8bit(const cv::Mat& in, cv::Mat& out)
	{
		if (in.channels() == 3)
		{
			// Make sure it's in a correct format
			if (in.depth() == CV_16U)
			{
				cv::Mat tmp = in / 256;
				tmp.convertTo(out, CV_8U);
				cv::cvtColor(out, out, CV_BGR2GRAY);
			}
			else
			{
				cv::cvtColor(in, out, CV_BGR2GRAY);
			}
		}
		else if (in.channels() == 4)
		{
			if (in.depth() == CV_16U)
			{
				cv::Mat tmp = in / 256;
				tmp.convertTo(out, CV_8U);
				cv::cvtColor(out, out, CV_BGRA2GRAY);
			}
			else
			{
				cv::cvtColor(in, out, CV_BGRA2GRAY);
			}
		}
		else
		{
			if (in.depth() == CV_16U)
			{
				cv::Mat tmp = in / 256;
				tmp.convertTo(out, CV_8U);
			}
			else if (in.depth() == CV_8U)
			{
				out = in.clone();
			}
		}
	}


}
#endif