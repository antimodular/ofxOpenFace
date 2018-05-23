///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// THIS SOFTWARE IS PROVIDED “AS IS” FOR ACADEMIC USE ONLY AND ANY EXPRESS
// OR IMPLIED WARRANTIES WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY.
// OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Notwithstanding the license granted herein, Licensee acknowledges that certain components
// of the Software may be covered by so-called “open source” software licenses (“Open Source
// Components”), which means any software licenses approved as open source licenses by the
// Open Source Initiative or any substantially similar licenses, including without limitation any
// license that, as a condition of distribution of the software licensed under such license,
// requires that the distributor make the software available in source code format. Licensor shall
// provide a list of Open Source Components for a particular version of the Software upon
// Licensee’s request. Licensee will comply with the applicable terms of such licenses and to
// the extent required by the licenses covering Open Source Components, the terms of such
// licenses will apply in lieu of the terms of this Agreement. To the extent the terms of the
// licenses applicable to Open Source Components prohibit any of the restrictions in this
// License Agreement with respect to such Open Source Component, such restrictions will not
// apply to such Open Source Component. To the extent the terms of the licenses applicable to
// Open Source Components require Licensor to make an offer to provide source code or
// related information in connection with the Software, such offer is hereby made. Any request
// for source code or related information should be directed to cl-face-tracker-distribution@lists.cam.ac.uk
// Licensee acknowledges receipt of notices for the Open Source Components for the initial
// delivery of the Software.

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

#ifndef __FACE_DETECTOR_MTCNN_h_
#define __FACE_DETECTOR_MTCNN_h_

// OpenCV includes
#include <opencv2/core/core.hpp>

// System includes
#include <vector>

using namespace std;

namespace LandmarkDetector
{
	class CNN
	{
	public:

		//==========================================

		// Default constructor
		CNN() { ; }

		// Copy constructor
		CNN(const CNN& other);

		// Given an image apply a CNN on it, the boolean direct controls if direct convolution is used (through matrix multiplication) or an FFT optimization
		std::vector<cv::Mat_<float> > Inference(const cv::Mat& input_img, bool direct = true, bool thread_safe = false);

		// Reading in the model
		void Read(const string& location);

		// Clearing precomputed DFTs
		void ClearPrecomp();

		size_t NumberOfLayers() { return cnn_layer_types.size(); }

	private:
		//==========================================
		// Convolutional Neural Network

		// CNN layers
		// Layer -> Weight matrix
		vector<cv::Mat_<float> > cnn_convolutional_layers_weights;

		// Keeping some pre-allocated im2col data as malloc is a significant time cost (not thread safe though)
		vector<cv::Mat_<float> > conv_layer_pre_alloc_im2col;

		// Layer -> kernel -> input maps
		vector<vector<vector<cv::Mat_<float> > > > cnn_convolutional_layers;
		vector<vector<float > > cnn_convolutional_layers_bias;
		// Layer matrix + bas
		vector<cv::Mat_<float> >  cnn_fully_connected_layers_weights;
		vector<cv::Mat_<float> > cnn_fully_connected_layers_biases;
		vector<cv::Mat_<float> >  cnn_prelu_layer_weights;
		vector<std::tuple<int, int, int, int> > cnn_max_pooling_layers;

		// Precomputations for faster convolution
		vector<vector<map<int, vector<cv::Mat_<double> > > > > cnn_convolutional_layers_dft;

		// CNN: 0 - convolutional, 1 - max pooling, 2 - fully connected, 3 - prelu, 4 - sigmoid
		vector<int > cnn_layer_types;
	};
	//===========================================================================
	//
	// Checking if landmark detection was successful using an SVR regressor
	// Using multiple validators trained add different views
	// The regressor outputs -1 for ideal alignment and 1 for worst alignment
	//===========================================================================
	class FaceDetectorMTCNN
	{

	public:

		// Default constructor
		FaceDetectorMTCNN() { ; }

		FaceDetectorMTCNN(const string& location);

		// Copy constructor
		FaceDetectorMTCNN(const FaceDetectorMTCNN& other);

		// Given an image, orientation and detected landmarks output the result of the appropriate regressor
		bool DetectFaces(vector<cv::Rect_<float> >& o_regions, const cv::Mat& input_img, std::vector<float>& o_confidences, int min_face = 60, float t1 = 0.6, float t2 = 0.7, float t3 = 0.7);

		// Reading in the model
		void Read(const string& location);

		// Indicate if the model has been read in
		bool empty() { return PNet.NumberOfLayers() == 0 || RNet.NumberOfLayers() == 0 || ONet.NumberOfLayers() == 0; };

	private:
		//==========================================
		// Components of the model

		CNN PNet;
		CNN RNet;
		CNN ONet;
		
	};

}
#endif
