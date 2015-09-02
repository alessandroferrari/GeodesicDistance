/*
 * dtocs.hpp
 *
 *  Created on: Sep 2, 2015
 *      Author: alessandro
 */

#ifndef DTOCS_HPP_
#define DTOCS_HPP_

#include "opencv2/core/core.hpp"
#include "geodesic_distance.hpp"

double DTOCS_diff(double ge, double gdiff, Params params, bool variant);
cv::Mat DTOCS(cv::Mat& graymap, cv::Mat& mask, bool border_correct = false, double alpha = 1.0);

#endif /* DTOCS_HPP_ */
