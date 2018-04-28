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

#ifndef __ROTATION_HELPERS_h_
#define __ROTATION_HELPERS_h_

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>

namespace Utilities
{
	//===========================================================================
	// Angle representation conversion helpers
	//===========================================================================

	// Using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
	static cv::Matx33d Euler2RotationMatrix(const cv::Vec3d& eulerAngles)
	{
		cv::Matx33d rotation_matrix;

		double s1 = sin(eulerAngles[0]);
		double s2 = sin(eulerAngles[1]);
		double s3 = sin(eulerAngles[2]);

		double c1 = cos(eulerAngles[0]);
		double c2 = cos(eulerAngles[1]);
		double c3 = cos(eulerAngles[2]);

		rotation_matrix(0, 0) = c2 * c3;
		rotation_matrix(0, 1) = -c2 *s3;
		rotation_matrix(0, 2) = s2;
		rotation_matrix(1, 0) = c1 * s3 + c3 * s1 * s2;
		rotation_matrix(1, 1) = c1 * c3 - s1 * s2 * s3;
		rotation_matrix(1, 2) = -c2 * s1;
		rotation_matrix(2, 0) = s1 * s3 - c1 * c3 * s2;
		rotation_matrix(2, 1) = c3 * s1 + c1 * s2 * s3;
		rotation_matrix(2, 2) = c1 * c2;

		return rotation_matrix;
	}

	// Using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
	static cv::Vec3d RotationMatrix2Euler(const cv::Matx33d& rotation_matrix)
	{
		double q0 = sqrt(1 + rotation_matrix(0, 0) + rotation_matrix(1, 1) + rotation_matrix(2, 2)) / 2.0;
		double q1 = (rotation_matrix(2, 1) - rotation_matrix(1, 2)) / (4.0*q0);
		double q2 = (rotation_matrix(0, 2) - rotation_matrix(2, 0)) / (4.0*q0);
		double q3 = (rotation_matrix(1, 0) - rotation_matrix(0, 1)) / (4.0*q0);

		double t1 = 2.0 * (q0*q2 + q1*q3);

		double yaw = asin(2.0 * (q0*q2 + q1*q3));
		double pitch = atan2(2.0 * (q0*q1 - q2*q3), q0*q0 - q1*q1 - q2*q2 + q3*q3);
		double roll = atan2(2.0 * (q0*q3 - q1*q2), q0*q0 + q1*q1 - q2*q2 - q3*q3);

		return cv::Vec3d(pitch, yaw, roll);
	}

	static cv::Vec3d Euler2AxisAngle(const cv::Vec3d& euler)
	{
		cv::Matx33d rotMatrix = Euler2RotationMatrix(euler);
		cv::Vec3d axis_angle;
		cv::Rodrigues(rotMatrix, axis_angle);
		return axis_angle;
	}

	static cv::Vec3d AxisAngle2Euler(const cv::Vec3d& axis_angle)
	{
		cv::Matx33d rotation_matrix;
		cv::Rodrigues(axis_angle, rotation_matrix);
		return RotationMatrix2Euler(rotation_matrix);
	}

	static cv::Matx33d AxisAngle2RotationMatrix(const cv::Vec3d& axis_angle)
	{
		cv::Matx33d rotation_matrix;
		cv::Rodrigues(axis_angle, rotation_matrix);
		return rotation_matrix;
	}

	static cv::Vec3d RotationMatrix2AxisAngle(const cv::Matx33d& rotation_matrix)
	{
		cv::Vec3d axis_angle;
		cv::Rodrigues(rotation_matrix, axis_angle);
		return axis_angle;
	}

	// Generally useful 3D functions
	static void Project(cv::Mat_<double>& dest, const cv::Mat_<double>& mesh, double fx, double fy, double cx, double cy)
	{
		dest = cv::Mat_<double>(mesh.rows, 2, 0.0);

		int num_points = mesh.rows;

		double X, Y, Z;

		cv::Mat_<double>::const_iterator mData = mesh.begin();
		cv::Mat_<double>::iterator projected = dest.begin();

		for (int i = 0; i < num_points; i++)
		{
			// Get the points
			X = *(mData++);
			Y = *(mData++);
			Z = *(mData++);

			double x;
			double y;

			// if depth is 0 the projection is different
			if (Z != 0)
			{
				x = ((X * fx / Z) + cx);
				y = ((Y * fy / Z) + cy);
			}
			else
			{
				x = X;
				y = Y;
			}

			// Project and store in dest matrix
			(*projected++) = x;
			(*projected++) = y;
		}

	}

}
#endif