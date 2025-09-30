cpv: alglib.a classMember.h modelState.h modelState.cpp fit.cpp fit.h main.cpp process.cpp process.h test.cpp test.h \
CSVWrite.hpp nanoflann/KDTreeVectorOfVectorsAdaptor.h nanoflann/nanoflann.hpp nanoflann/WeightedL2MetricAdaptor.hpp \
cachePaths.h runFromCache.cpp runFromCache.h runFull.cpp runFull.h
	g++ -Ialglib/src -std=c++11 -pthread -o cpv fit.cpp main.cpp process.cpp test.cpp modelState.cpp runFromCache.cpp \
	runFull.cpp cachePaths.cpp nanoflann/KDTreeVectorOfVectorsAdaptor.h nanoflann/nanoflann.hpp nanoflann/WeightedL2MetricAdaptor.hpp alglib.a
alglib.a:
	cd alglib/src && $(MAKE)

clean:
	find . -name '*.[oa]' -print | xargs /bin/rm -f
