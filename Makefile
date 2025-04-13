cpv: alglib.a classMember.h modelState.h fit.cpp fit.h main.cpp process.cpp process.h test.cpp test.h CSVWrite.hpp
	g++ -Ialglib/src -std=c++11 -pthread -o cpv fit.cpp main.cpp process.cpp test.cpp alglib.a

alglib.a:
	cd alglib/src && $(MAKE)

clean:
	find . -name '*.[oa]' -print | xargs /bin/rm -f
