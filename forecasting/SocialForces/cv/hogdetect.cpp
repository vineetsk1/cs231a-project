#include "mex.h"
#include "cxcore.h"
#include "cvaux.h"
using namespace cv;


/*
 * hogdetect - mex wrapper for opencv HOG
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
  if(nrhs!=1 || nlhs!=1) mexErrMsgTxt("Number of arguments incorrect");
  
  /* Detect person with HOG+SVM detector */
  Mat* img = cvmxArrayToMat(prhs[0],CV_8U);
  cvtColor(*img,*img,CV_RGB2BGR);

  HOGDescriptor hog;
  vector<Rect> found;
  hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
  hog.detectMultiScale(*img, found, 0, Size(8,8), Size(32,32), 1.05, 2);

  /* Reshape the result*/
  Mat result(Size(4,found.size()),CV_32SC1);
  for (size_t i = 0; i < found.size(); i++) {
	  result.at<int>(i,0) = found[i].x;
	  result.at<int>(i,1) = found[i].y;
	  result.at<int>(i,2) = found[i].width;
	  result.at<int>(i,3) = found[i].height;
  }
  plhs[0] = cvmxArrayFromMat(&result,mxDOUBLE_CLASS);

  /* Clean up */
  delete img;
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

