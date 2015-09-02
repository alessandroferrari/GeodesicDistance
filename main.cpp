#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "wdtocs.hpp"
#include "dtocs.hpp"

using cv::Mat;
using cv::Size;
using cv::imread;
using cv::imwrite;
using cv::BORDER_DEFAULT;

int main(){

	Mat img = imread("img.jpg");
	Size s = img.size();
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	Mat tmp = imread("mask.png");
	Mat mask;
	cvtColor(tmp,mask, CV_BGR2GRAY);
	
	Mat grad;
 	int scale = 1;
 	int delta = 0;
 	int ddepth = CV_16S;

	 /// Generate grad_x and grad_y
  	Mat grad_x, grad_y;
  	Mat abs_grad_x, abs_grad_y;

	Sobel( gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  	convertScaleAbs( grad_x, abs_grad_x );	

	Sobel( gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  	convertScaleAbs( grad_y, abs_grad_y );

  	/// Total Gradient (approximate)
  	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

  	imwrite("grad.jpg",grad);


	Mat result = DTOCS(grad,mask);
	imwrite("DTOCS.png",result);


	Mat new_result = WDTOCS(grad,mask);
	imwrite("WDTOCS.png",new_result);

}
