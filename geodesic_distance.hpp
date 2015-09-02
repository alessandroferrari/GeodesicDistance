/*
 * geodesic_distance.hpp
 *
 *  Created on: Sep 2, 2015
 *      Author: alessandro
 */

#ifndef GEODESIC_DISTANCE_HPP_
#define GEODESIC_DISTANCE_HPP_

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

struct Params{
	double alpha;
	double beta;
	double delta;
};

template <typename T> cv::Mat geodesic_distance(cv::Mat& graymap, cv::Mat& mask, double (&diff)(double,double,Params,bool), Params params, bool border_correct = true){

    cv::Size s = graymap.size();

    double maxint = (double) std::numeric_limits<int32_t>::max();
    cv::Mat distance = cv::Mat::zeros(s.height, s.width, CV_64FC1);
    double *dist = (double*) distance.data ;
    uint8_t *m = (uint8_t*) mask.data;
    T *gmap = (T*) graymap.data;

    //mask initialization
    for(int i=0;i<s.height;i++){
        for(int j=0;j<s.width;j++){
            if(!m[i*s.width+j]){
                dist[i*s.width+j] = maxint;
            }else{
                dist[i*s.width+j] = 0;
            }
        }
    }

    int a,b,c,d,e,f,g,h,k;
    double da,db,dc,dd,df,dg,dh,dk;
    double fa,fb,fc,fd,ff,fg,fh,fk;
    int stride, stride_m1, stride_p1;
    double tmp;

    //first pass
    if(border_correct){
		//first column
		for(int i=1;i<s.height;i++){
			stride = i * s.width;
			stride_m1 = (i-1) * s.width;
			b = stride_m1;
			e = stride;
			c = stride_m1 + 1;

			db = diff(gmap[e],gmap[b],params,false);
			dc = diff(gmap[e],gmap[c],params,true);

			fb = dist[b]+db;
			fc = dist[c]+dc;
			tmp = (fb>fc)?fc:fb;
			dist[e] = (dist[e]>tmp)?tmp:dist[e];
		}

		//first line
		for(int j=1;j<s.width;j++){
			d = j - 1;
			e = j;

			dd = diff(gmap[e],gmap[d],params,false);

			fd = dist[d]+dd;
			dist[e] = (dist[e]>fd)?fd:dist[e];
		}
    }

    //the grid
    for(int i=1;i<s.height;i++){
        for(int j=1;j<s.width-1;j++){
            stride = i * s.width;
            stride_m1 = (i-1) * s.width;
            a = stride_m1 + j - 1;
            b = stride_m1 + j;
            c = stride_m1 + j + 1;
            d = stride + j - 1;
            e = stride + j;
            da = diff(gmap[e],gmap[a],params,true);
            db = diff(gmap[e],gmap[b],params,false);
            dc = diff(gmap[e],gmap[c],params,true);
            dd = diff(gmap[e],gmap[d],params,false);

            fa = dist[a]+da;
            fb = dist[b]+db;
            fc = dist[c]+dc;
            fd = dist[d]+dd;
            tmp = (fa > fb)?fb:fa;
            tmp = (tmp > fc)?fc:tmp;
            tmp = (tmp > fd)?fd:tmp;
            tmp = (tmp > dist[e])?dist[e]:tmp;
            dist[e] = tmp;
        }
    }

    if(border_correct){
		//last column
		for(int i=1;i<s.height;i++){
			stride = i * s.width;
			stride_m1 = (i-1) * s.width;
			e = stride + s.width - 1;
			a = stride_m1 + s.width - 2;
			b = stride_m1 + s.width - 1;
			d = stride_m1 + s.width - 2;

			da = diff(gmap[e],gmap[a],params,true);
			db = diff(gmap[e],gmap[b],params,false);
			dd = diff(gmap[e],gmap[d],params,false);

			fa = dist[a]+da;
			fb = dist[b]+db;
			fd = dist[d]+dd;

			tmp = (fa > fb)?fb:fa;
			tmp = (tmp > fd)?fd:tmp;
			tmp = (tmp > dist[e])?dist[e]:tmp;
			dist[e] = tmp;
		}
    }
    //second pass

    if(border_correct){
		//last line
		for(int j=s.width-2;j>=0;j--){
			stride = s.width*(s.height-2);
			f = stride + (j+1);
			e = stride + j;

			df = diff(gmap[e],gmap[f],params,false);

			ff = dist[f]+df;
			dist[e] = (dist[e]>ff)?ff:dist[e];
		}

		//last column
		for(int i=s.height-2;i>=0;i--){
			stride_p1 = (i+1)*s.width;
			stride = i*s.width;
			h = stride_p1;
			g = stride_p1-1;
			e = stride ;

			dh = diff(gmap[e],gmap[h],params,false);
			dg = diff(gmap[e],gmap[g],params,true);

			fh = dist[h]+dh;
			fg = dist[g]+dg;
			tmp = (fh>fg)?fg:fh;
			dist[e] = (dist[e]>tmp)?tmp:dist[e];
		}
    }

    for(int i=s.height-2; i>=0;i--){
        for(int j=s.width-2; j>0;j--){
            stride_p1 = (i+1)*s.width;
            stride = i*s.width;
            e = stride + j;
            f = stride + j + 1;
            g = stride_p1 + j - 1;
            h = stride_p1 + j;
            k = stride_p1 + j + 1;

            df = diff(gmap[e],gmap[f],params,false);
            dg = diff(gmap[e],gmap[g],params,true);
            dh = diff(gmap[e],gmap[h],params,false);
            dk = diff(gmap[e],gmap[k],params,true);
            ff = dist[f]+df;
            fg = dist[g]+dg;
            fk = dist[k]+dk;
            fh = dist[h]+dh;
            tmp = (fh>fg)?fg:fh;
            tmp = (fk>tmp)?tmp:fk;
            tmp = (ff>tmp)?tmp:ff;

            dist[e] = (dist[e]>tmp)?tmp:dist[e];
        }
    }

    if(border_correct){
		//first column
		for(int i=s.height-2;i>0;i++){
			stride_p1 = (i+1)*s.width;
			stride = i*s.width;
			e = stride ;
			f = stride + 1;
			h = stride_p1;
			k = stride_p1 + 1;

			df = diff(gmap[e],gmap[f],params,false);
			dh = diff(gmap[e],gmap[h],params,false);
			dk = diff(gmap[e],gmap[k],params,true);

			ff = dist[f]+df;
			fk = dist[k]+dk;
			fh = dist[h]+dh;
			tmp = (fh>fk)?fk:fh;
			tmp = (ff>tmp)?tmp:ff;

			dist[e] = (dist[e]>tmp)?tmp:dist[e];
		}
    }

    return distance;
}

#endif /* GEODESIC_DISTANCE_HPP_ */
