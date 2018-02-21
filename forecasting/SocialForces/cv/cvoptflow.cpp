#include "mex.h"
#include "cv.h"
using namespace cv;

/*
 * cvtrack - matlab mex wrapper for opencv
 *
 */
 
mxArray* cvmxArrayFromMat(const Mat *mat, mxClassID classid=mxUNKNOWN_CLASS);
Mat* cvmxArrayToMat(const mxArray *arr, int depth=CV_USRTYPE1);
int cvmxClassIdToMatDepth(mxClassID classid);
mxClassID cvmxClassIdFromMatDepth(int depth);

/*
 * mexFunction: main routine called from Matlab
 */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
	Size winSize(15,15);
	/* Check for proper number of arguments. */
	if(nrhs<3) mexErrMsgTxt("Number of arguments incorrect");
	/* Catch Template size */
	if(nrhs>=5&&mxIsDouble(prhs[4])&&mxGetM(prhs[4])==1&&mxGetN(prhs[4])==2) {
		double *p = mxGetPr(prhs[4]);
		winSize.width = p[0]; winSize.height = p[1];
	}

	/* Convert data format */
	Mat* img1 = cvmxArrayToMat(prhs[0]);	/* Image at time t-1 */
	Mat* img2 = cvmxArrayToMat(prhs[1]);	/* Image at time t */
	Mat* x1 = cvmxArrayToMat(prhs[2],CV_32S);	/* Position (x,y) at t */
	Mat* x2 = 0;
	if (nrhs>=4) {
		x2 = cvmxArrayToMat(prhs[3],CV_32S);	/* Velocity (dx,dy) at t */
	} else {
		x2 = new Mat();
		*x2 = x1->clone();
	}

	// Copy points
	vector<Point2f> pts1(x1->cols,Point2f(0,0));
	vector<Point2f> pts2(x1->cols,Point2f(0,0));
	for (int i = 0; i < x1->cols; i++) {
		pts1.at(i).x = x1->at<int>(0,i) - 1;
		pts1.at(i).y = x1->at<int>(1,i) - 1;
		pts2.at(i).x = x2->at<int>(0,i) - 1;
		pts2.at(i).y = x2->at<int>(1,i) - 1;
	}

	// Calculate optical flow
	vector<uchar> status;
	vector<float> error;
	calcOpticalFlowPyrLK(*img1,				// previous frame
			*img2,							// next frame
			pts1,							// points at previous frame
			pts2,							// returned points at next frame
			status,							// convergence status
			error,							// error at convergence
			winSize,						// window size
			3,								// maxlevel
			TermCriteria( TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
			0.5,							// derivLambda
			OPTFLOW_USE_INITIAL_FLOW);		// flag

	// Copy points
	for (int i = 0; i < x1->cols; i++) {
		x2->at<int>(0,i) = pts2.at(i).x + 1;
		x2->at<int>(1,i) = pts2.at(i).y + 1;
	}

	// Return
	plhs[0] = cvmxArrayFromMat(x2,mxDOUBLE_CLASS);

	/* Clean up */
	delete img1;
	delete img2;
	delete x1;
	delete x2;
}

/*
 * cvMatType
 */
int cvmxClassIdToMatDepth(mxClassID classid) {
	int depth;
	switch(classid) {
	case mxDOUBLE_CLASS:	depth = CV_64F;	break;
	case mxSINGLE_CLASS:	depth = CV_32F;	break;
	case mxINT8_CLASS:		depth = CV_8S;	break;
	case mxUINT8_CLASS:		depth = CV_8U;	break;
	case mxINT16_CLASS:		depth = CV_16S;	break;
	case mxUINT16_CLASS:	depth = CV_16U;	break;
	case mxINT32_CLASS:		depth = CV_32S;	break;
	default: 				depth = CV_USRTYPE1; break;
	}
	return depth;
}

mxClassID cvmxClassIdFromMatDepth(int depth) {
	mxClassID classid;
	switch(depth) {
	case CV_64F:	classid = mxDOUBLE_CLASS;		break;
	case CV_32F:	classid = mxSINGLE_CLASS;		break;
	case CV_8S:		classid = mxINT8_CLASS;			break;
	case CV_8U:		classid = mxUINT8_CLASS;		break;
	case CV_16S:	classid = mxINT16_CLASS;		break;
	case CV_16U:	classid = mxUINT16_CLASS;		break;
	case CV_32S:	classid = mxINT32_CLASS;		break;
	default: 		classid = mxUNKNOWN_CLASS;		break;
	}
	return classid;
}

/*
 * cvmxArrayToMat: converts mxArray to cv::Mat
 * @arr: mxArray object
 * @depth: destination datatype, e.g., CV_8U, CV_32F
 * @return: cv::Mat object
 */
Mat* cvmxArrayToMat(const mxArray *arr, int depth) {
	mwSize nDim = mxGetNumberOfDimensions(arr);
	const mwSize *dims = mxGetDimensions(arr);
	mwSize subs[] = {0,0,0};
	int nChannels = (nDim > 2) ? dims[2] : 1;
	if(depth == CV_USRTYPE1) depth = cvmxClassIdToMatDepth(mxGetClassID(arr));

	/* Create cv::Mat object */
	/* Note that new mat is transposed      */
	/* To fix it, call mat->t() when needed */
	/* This however requires copying data   */
	Mat *mat = new Mat(dims[1],dims[0],CV_MAKETYPE(depth,nChannels));
	Mat mv[nChannels];
	/* Copy each channel */
	for (int i = 0; i<nChannels; i++) {
		subs[2] = i;
		void *ptr = (void*)((mwIndex)mxGetData(arr)+
				mxGetElementSize(arr)*mxCalcSingleSubscript(arr,3,subs));
		Mat m(mat->rows,mat->cols,
				CV_MAKETYPE(cvmxClassIdToMatDepth(mxGetClassID(arr)),1),
				ptr,mxGetElementSize(arr)*mat->cols);
		m.convertTo(mv[i],CV_MAKETYPE(depth,1)); /* Read from mxArray through m */
	}
	merge(mv,nChannels,*mat);
	return mat;
}

/*
 * cvmxArrayFromMat: converts cv::Mat to mxArray
 * @mat: cv::Mat object
 * @return: mxArray object
 */
mxArray* cvmxArrayFromMat(const Mat *mat, mxClassID classid) {
	mwSize nDim = (mat->channels() > 1) ? 3 : 2;
	mwSize dims[3], subs[] = {0,0,0};
	dims[0] = mat->cols;
	dims[1] = mat->rows;
	int nChannels = mat->channels();
	if (nDim > 2) dims[2] = nChannels;
	if (classid == mxUNKNOWN_CLASS) classid = cvmxClassIdFromMatDepth(mat->depth());
	int type = CV_MAKETYPE(cvmxClassIdToMatDepth(classid),1); /* destination type */

	/* Allocate memory */
	mxArray *arr = mxCreateNumericArray(nDim,dims,classid,mxREAL);
	vector<Mat> mv;
	split(*mat,mv);
	/* Copy each channel */
	for (int i = 0; i < nChannels; i++) {
		subs[2] = i;
		void *ptr = (void*)((mwIndex)mxGetData(arr)+
				mxGetElementSize(arr)*mxCalcSingleSubscript(arr,3,subs));
		Mat m(mat->rows,mat->cols,type,ptr,mxGetElementSize(arr)*mat->cols);
		mv.at(i).convertTo(m,type); /* Write to mxArray through m */
	}
	return arr;
}
