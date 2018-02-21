#include "mex.h"
#include "cv.h"
#include "highgui.h"
using namespace cv;

/*
 * cvread - matlab mex wrapper for opencv
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
	/* Check for proper number of arguments. */
	if(nrhs<2 || nlhs!=1) mexErrMsgTxt("Number of arguments incorrect");

	/* Convert data format */
	string filename = String((char*)mxGetData(prhs[0]),mxGetN(prhs[0]));
	double* f = mxGetPr(prhs[1]);	/* Frame index */
	Mat frame, img;

	VideoCapture vid(filename);
	if (!vid.isOpened()) mexErrMsgTxt("Unable to open video file");
	vid.set(CV_CAP_PROP_POS_FRAMES,f[0]);
	vid >> frame;
	cvtColor(frame,img,CV_BGR2RGB);

	/* Convert data format */
	plhs[0] = cvmxArrayFromMat(&img,mxUINT8_CLASS);
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

	/* Allocate memory */
	Mat *mat = new Mat(dims[0],dims[1],CV_MAKETYPE(depth,nChannels));
	Mat mv[nChannels];
	/* Copy each channel */
	for (int i = 0; i<nChannels; i++) {
		subs[2] = i;
		void *ptr = (void*)((mwIndex)mxGetData(arr)+
				mxGetElementSize(arr)*mxCalcSingleSubscript(arr,3,subs));
		Mat m(dims[1],dims[0],
				CV_MAKETYPE(cvmxClassIdToMatDepth(mxGetClassID(arr)),1),
				ptr,mxGetElementSize(arr)*dims[0]);
		Mat mt = m.t();
		mt.convertTo(mv[i],CV_MAKETYPE(depth,1)); /* Read from mxArray through m */
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
	dims[0] = mat->size().height;
	dims[1] = mat->size().width;
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
		Mat m(dims[1],dims[0],type,ptr,mxGetElementSize(arr)*dims[0]);
		Mat mt = mv[i].t();
		mt.convertTo(m,type); /* Write to mxArray through m */
	}
	return arr;
}

