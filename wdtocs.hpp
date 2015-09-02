/*
 * wdtocs.hpp
 *
 *  Created on: Sep 2, 2015
 *      Author: alessandro
 */

#ifndef WDTOCS_HPP_
#define WDTOCS_HPP_

#include "opencv2/core/core.hpp"
#include "geodesic_distance.hpp"

double WDTOCS_diff(double ge, double gdiff, Params params, bool variant);
cv::Mat WDTOCS(cv::Mat& graymap, cv::Mat& mask, bool border_correct = false, double alpha = 1.0, double beta = 1.36930, double delta = 0.95509);

#endif /* WDTOCS_HPP_ */
