#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

ifeq ($(multiproc), 1)
	MP = -fopenmp -Dmultiproc
endif
ifeq ($(debug), 1)
	DBG = -g
endif
ifeq ($(debugMat), 1)
	DBG = -g -Ddbgmat
endif
ifeq ($(optim), 1)
	OPT = -O2
endif

CF = -lgsl -lgslcblas -lm $(MP) $(DBG)

all: genDirStructure convertFPFBinary convertPropBinary svmHyper knnHyper nbHyper seaHyper1 seaHyper2 seaHyper3 clusterMinFull

genDirStructure:
	g++ genDirStructure.cpp -o genDirStructure.exec $(CF)
convertFPFBinary:
	g++ convertFPFBinary.cpp -o convertFPFBinary.exec $(CF)
convertPropBinary:
	g++ convertPropBinary.cpp -o convertPropBinary.exec $(CF)
svmHyper:
	$(shell ln -s  ${HOME}/myprogs/lib/libsvm-dense-3.17Modified libsvm)
	make -C ${HOME}/myprogs/lib/libsvm-dense-3.17Modified svm.o
	g++ svmHyper.cpp targetPipeline.cpp sums.cpp ${HOME}/myprogs/lib/libsvm-dense-3.17Modified/svm.o -o svmHyper.exec -std=c++11 -lconfig++ $(CF)
knnHyper:
	g++ knnHyper.cpp targetPipeline.cpp sums.cpp -o knnHyper.exec -std=c++11 -lconfig++ $(CF)
nbHyper:
	g++ nbHyper.cpp targetPipeline.cpp sums.cpp -o nbHyper.exec -std=c++11 -lconfig++ $(CF)
seaHyper1:
	g++ seaHyper1.cpp targetPipeline.cpp sums.cpp -o seaHyper1.exec -std=c++11 -lgsl -lconfig++ $(CF)
seaHyper2:
	g++ seaHyper2.cpp targetPipeline.cpp sums.cpp -o seaHyper2.exec -I${HOME}/myprogs/lib/dlib-19.0 -std=c++11 -lgsl -lconfig++ $(CF)
seaHyper3:
	g++ seaHyper3.cpp targetPipeline.cpp sums.cpp -o seaHyper3.exec -std=c++11 -lconfig++ $(CF)
clusterMinFull:
	g++ clusterMinFull.cpp -o clusterMinFull.exec -std=c++11 $(MP)