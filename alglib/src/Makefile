CXX=g++ -std=c++11

../../alglib.a: alglibinternal.o alglibmisc.o ap.o dataanalysis.o diffequations.o fasttransforms.o integration.o interpolation.o kernels_avx2.o kernels_fma.o kernels_sse2.o linalg.o optimization.o solvers.o specialfunctions.o statistics.o
	ar r ../../alglib.a *.o

alglibinternal.o: alglibinternal.cpp
alglibmisc.o: alglibmisc.cpp
ap.o: ap.cpp
dataanalysis.o: dataanalysis.cpp
diffequations.o: diffequations.cpp
fasttransforms.o: fasttransforms.cpp
integration.o: integration.cpp
interpolation.o: interpolation.cpp
kernels_avx2.o: kernels_avx2.cpp
kernels_fma.o: kernels_fma.cpp
kernels_sse2.o: kernels_sse2.cpp
linalg.o: linalg.cpp
optimization.o: optimization.cpp
solvers.o: solvers.cpp
specialfunctions.o: specialfunctions.cpp
statistics.o: statistics.cpp
