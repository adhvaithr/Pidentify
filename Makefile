cpv: alglib.a classMember.h fit.cpp fit.h main.cpp process.cpp process.h
	g++ -Ialglib/src -std=c++11 -o cpv fit.cpp main.cpp process.cpp alglib.a

alglib.a:
	cd alglib/src && $(MAKE)
