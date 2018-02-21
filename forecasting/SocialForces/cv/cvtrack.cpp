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
	Point tmlSize(8,8);
	Point roiSize(16,16);

	/* Check for proper number of arguments. */
	if(nrhs<4 || nlhs!=2) mexErrMsgTxt("Number of arguments incorrect");
	/* Catch Template size */
	if(nrhs>=5&&mxIsDouble(prhs[4])&&mxGetM(prhs[4])==1&&mxGetN(prhs[4])==2) {
		double *p = mxGetPr(prhs[4]);
		tmlSize.y = p[0]; tmlSize.x = p[1];
	}
	/* Catch ROI size */
	if(nrhs>=6&&mxIsDouble(prhs[5])&&mxGetM(prhs[5])==1&&mxGetN(prhs[5])==2) {
		double *p = mxGetPr(prhs[5]);
		roiSize.y = p[0]; roiSize.x = p[1];
	}
	Point mapSize(((roiSize.x-tmlSize.x)<<1)+1,((roiSize.y-tmlSize.y)<<1)+1);
	Point mapOffset((mapSize.x-1)/2,(mapSize.y-1)/2);
	int nMap = mapSize.x*mapSize.y;

	/* Convert data format */
	Mat* img1 = cvmxArrayToMat(prhs[0],CV_8U);	/* Image at time t-1 */
	Mat* img2 = cvmxArrayToMat(prhs[1],CV_8U);	/* Image at time t */
	Mat* x = cvmxArrayToMat(prhs[2],CV_32S);	/* Position (x,y) at t-1 */
	Mat* xh = cvmxArrayToMat(prhs[3],CV_32S);	/* Position (x,y) at t */
	Mat result(nMap*x->rows,4,CV_64F);	/* Table (id,px,py,score) to return */

	/* Apply matchTemplate for each subject */
	int j = 0;
	for (int i = 0; i < x->rows; i++) {
		/* Sample model patch */
		Mat tml = img1->rowRange(x->at<int>(i,0)-tmlSize.y,x->at<int>(i,0)+tmlSize.y)
					   .colRange(x->at<int>(i,1)-tmlSize.x,x->at<int>(i,1)+tmlSize.x);
		/* Specify ROI */
		Mat roi = img2->rowRange(xh->at<int>(i,0)-roiSize.y,xh->at<int>(i,0)+roiSize.y)
					   .colRange(xh->at<int>(i,1)-roiSize.x,xh->at<int>(i,1)+roiSize.x);
		/* Apply convolution */
		Mat map;
		matchTemplate(roi,tml,map,CV_TM_CCORR_NORMED); /*CV_TM_SQDIFF_NORMED*/
		/* Save the result */
		/* Create index structure */
		result.col(0).rowRange(j,j+nMap) = Scalar(i+1);
		for (int jy = 0; jy < mapSize.y; jy++) {
			for (int jx = 0; jx < mapSize.x; jx++) {
				result.at<double>(j+jx+jy*mapSize.x,1) = (double)((int)xh->at<int>(i,0)+jy-mapOffset.y);
				result.at<double>(j+jx+jy*mapSize.x,2) = (double)((int)xh->at<int>(i,1)+jx-mapOffset.x);
				result.at<double>(j+jx+jy*mapSize.x,3) = (double)(map.at<float>(jy,jx));
			}
		}
		j+=nMap;
		/* Min location */
		Point loc;
		minMaxLoc(map,0,0,0,&loc); /* min or max */
		xh->at<int>(i,0) += loc.y - mapOffset.y;
		xh->at<int>(i,1) += loc.x - mapOffset.x;
	}

	/* Convert data format */
	plhs[0] = cvmxArrayFromMat(xh,mxDOUBLE_CLASS);
	plhs[1] = cvmxArrayFromMat(&result,mxDOUBLE_CLASS);

	/* Clean up */
	delete img1;
	delete img2;
	delete x;
	delete xh;
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

