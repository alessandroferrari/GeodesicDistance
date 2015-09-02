/*
 * dtocs.cpp
 *
 *  Created on: Sep 2, 2015
 *      Author: alessandro
 */

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdint.h>
#include <cmath>
#include "geodesic_distance.hpp"
#include "dtocs.hpp"

using cv::Mat;
using std::abs;

double DTOCS_diff(double ge, double gdiff, Params params, bool variant){
	return params.alpha * abs(ge-gdiff);
}

cv::Mat DTOCS(cv::Mat& graymap, cv::Mat& mask, bool border_correct, double alpha){

	assert(graymap.channels()==1);
	assert(mask.channels()==1);
	assert(mask.depth()==CV_8U);

	Params params;
	params.alpha = alpha;

	int d = graymap.depth();
	Mat result;

	if(d==CV_8U){
		result = geodesic_distance<uint8_t>(graymap, mask, DTOCS_diff, params, border_correct);
	}else if(d==CV_8S){
		result = geodesic_distance<int8_t>(graymap, mask, DTOCS_diff, params, border_correct);
	}else if(d==CV_16U){
		result = geodesic_distance<uint16_t>(graymap, mask, DTOCS_diff, params, border_correct);
	}else if(d==CV_16S){
		result = geodesic_distance<int16_t>(graymap, mask, DTOCS_diff, params, border_correct);
	}else if(d==CV_32S){
		result = geodesic_distance<int32_t>(graymap, mask, DTOCS_diff, params, border_correct);
	}else if(d==CV_32F){
		result = geodesic_distance<float>(graymap, mask, DTOCS_diff, params, border_correct);
	}else if(d==CV_64F){
		result = geodesic_distance<double>(graymap, mask, DTOCS_diff, params, border_correct);
	}

	return result;

}

