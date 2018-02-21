MATLABHOME?=/Applications/MATLAB_R2011a_Student.app
CC=g++
MEXEXT=$(shell $(MATLABHOME)/bin/mexext)
MEXARCH=$(subst mex,,$(MEXEXT))
CFLAGS=-Wall -O3 -I$(MATLABHOME)/extern/include -D MATLAB_MEX_FILE `pkg-config --cflags opencv`
LDFLAGS=-L$(MATLABHOME)/bin/$(MEXARCH) -lmx -lmex -lmat -dynamiclib `pkg-config --libs opencv`
EXECUTABLE=cvbgsubtract2.$(MEXEXT) cvtrack.$(MEXEXT) cvtrackSQ.$(MEXEXT) cvimproc.$(MEXEXT) cvoptflow.$(MEXEXT) cvbgsubtract.$(MEXEXT) cvread.$(MEXEXT)

all: $(EXECUTABLE)
	
hogdetect.$(MEXEXT): hogdetect.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) $< -o $@
cvimproc.$(MEXEXT): cvimproc.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) $< -o $@
cvtrack.$(MEXEXT): cvtrack.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) $< -o $@
cvtrackSQ.$(MEXEXT): cvtrackSQ.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) $< -o $@
cvoptflow.$(MEXEXT): cvoptflow.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) $< -o $@
cvbgsubtract.$(MEXEXT): cvbgsubtract.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) $< -o $@
cvbgsubtract2.$(MEXEXT): cvbgsubtract2.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) $< -o $@
cvread.$(MEXEXT): cvread.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) $< -o $@
clean:
	rm -rf *.o *.$(MEXEXT)
