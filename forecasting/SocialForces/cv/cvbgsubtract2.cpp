#include "mex.h"
#include "cv.h"
using namespace cv;


/*
 * cvbgsubtract2 - mex wrapper for opencv
 *
 */

vector<Mat> cvmxArrayToMatVec(const mxArray *arr);
mxArray* cvmxArrayFromMatVec(const vector<Mat>& mv);
Mat* cvmxArrayToMat(const mxArray *arr, int depth=CV_USRTYPE1);
mxArray* cvmxArrayFromMat(const Mat *mat, mxClassID classid=mxUNKNOWN_CLASS);
int cvmxClassIdToMatDepth(mxClassID classid);
mxClassID cvmxClassIdFromMatDepth(int depth);

/*
 * mexFunction: main routine called from Matlab
 */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
	/* Check for proper number of arguments. */
	if(nrhs<3||nlhs!=2) mexErrMsgTxt("Number of arguments incorrect");

	/* Set options */
	float thresh = 5.0;
	if(nrhs>3) thresh = (float)(*mxGetPr(prhs[3]));

	/* Create data struct for input */
	vector<Mat> src = cvmxArrayToMatVec(prhs[0]);
	vector<Mat> bg  = cvmxArrayToMatVec(prhs[1]);
	double* n = mxGetPr(prhs[2]);
	vector<Mat> dst(src.size());
	vector<Mat> bgBuf(src.size());

	/* Do image processing for each channel */
	for (unsigned int i = 0; i < src.size(); i++) {
		Mat diff,mask;								// Temporary float array
		medianBlur(src[i],dst[i],3);				// Remove noise and pass to destination
		bgBuf[i] = bg[i].clone();					// Allocate new background
		dst[i].convertTo(diff,CV_32F);				// Convert to float
		accumulateWeighted(diff,bgBuf[i],1/(*n));	// Update background
		absdiff(diff,bgBuf[i],diff);				// Compute absolute difference
		mask = diff<thresh;							// Compute mask
		Scalar b = mean(dst[i],mask);				// Compute background mean
		dst[i].setTo(b,mask);						// Apply threshold
	}

	/* Arrange return values */
	plhs[0] = cvmxArrayFromMatVec(dst);
	plhs[1] = cvmxArrayFromMatVec(bgBuf);
  
//  /* Convert mxArray to cv::Mat */
//  Mat* src = cvmxArrayToMat(prhs[0],CV_32F);
//  Mat* bg = cvmxArrayToMat(prhs[1],CV_32F);
//  double* n = mxGetPr(prhs[2]);
//
//  /* Apply cumulative running average */
//  accumulateWeighted(*src,*bg,1/(*n));
//  absdiff(*src,*bg,*src);
////  *src = (((*src - *bg) + 256.0)*0.5);	// [-510,510] approximately mapped to [0,255]
//
//  /* Convert cv::Mat to mxArray */
//  plhs[0] = cvmxArrayFromMat(src,mxUINT8_CLASS);
//  plhs[1] = cvmxArrayFromMat(bg,mxSINGLE_CLASS);
//
//  /* Clean up */
//  delete src;
//  delete bg;
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
 * cvmxArrayToMatVec: create vector<Mat> header for mxArray without memory allocation
 * @arr: mxArray object
 * @return: vector<cv::Mat> object
 */
vector<Mat> cvmxArrayToMatVec(const mxArray *arr) {
	mwSize nDim = mxGetNumberOfDimensions(arr);
	const mwSize *dims = mxGetDimensions(arr);
	mwSize subs[] = {0,0,0};
	unsigned int nChannels = (nDim > 2) ? dims[2] : 1;

	/* Create cv::Mat object */
	vector<Mat> mv(nChannels);
	/* Copy each channel */
	for (unsigned int i = 0; i<nChannels; i++) {
		subs[2] = i;
		void *ptr = (void*)((mwIndex)mxGetData(arr)+
				mxGetElementSize(arr)*mxCalcSingleSubscript(arr,3,subs));
		mv[i] = Mat(dims[1],dims[0],	// rows, cols
					CV_MAKETYPE(cvmxClassIdToMatDepth(mxGetClassID(arr)),1),
					ptr,mxGetElementSize(arr)*dims[0]);
	}
	return mv;
}

/*
 * cvmxArrayFromMatVec: create vector<Mat> header for mxArray without memory allocation
 * @arr: mxArray object
 * @return: vector<cv::Mat> object
 */
mxArray* cvmxArrayFromMatVec(const vector<Mat>& mv) {
	mwSize nDim = (mv.size() > 1) ? 3 : 2;
	mwSize dims[3], subs[] = {0,0,0};
	dims[0] = mv[0].cols;
	dims[1] = mv[0].rows;
	if (nDim > 2) dims[2] = mv.size();
	int type = CV_MAKE_TYPE(mv[0].depth(),1);
	mxArray *arr = mxCreateNumericArray(nDim,dims,
						cvmxClassIdFromMatDepth(mv[0].depth()),mxREAL);
	/* Copy each channel */
	for (unsigned int i = 0; i < mv.size(); i++) {
		subs[2] = i;
		void *ptr = (void*)((mwIndex)mxGetData(arr)+
				mxGetElementSize(arr)*mxCalcSingleSubscript(arr,3,subs));
		Mat m(mv[i].rows,mv[i].cols,type,
				ptr,mxGetElementSize(arr)*mv[i].cols);
		mv[i].convertTo(m,type); /* Write to mxArray through m */
	}
	return arr;
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

