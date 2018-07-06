#!/bin/bash

#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

mkdir -p $HOME/myprogs
mkdir -p $HOME/myprogs/lib
mkdir -p $HOME/myprogs/source

wget https://src.fedoraproject.org/repo/pkgs/libconfig/libconfig-1.5.tar.gz/a939c4990d74e6fc1ee62be05716f633/libconfig-1.5.tar.gz --directory-prefix=$HOME/myprogs/source
wget http://dlib.net/files/dlib-19.0.zip --directory-prefix=$HOME/myprogs/source
wget http://www.bioinf.jku.at/research/lsc/libsvm-dense-3.17Modified.zip --directory-prefix=$HOME/myprogs/source

tar -xzvf $HOME/myprogs/source/libconfig-1.5.tar.gz --directory=$HOME/myprogs/source
unzip $HOME/myprogs/source/dlib-19.0.zip -d $HOME/myprogs/source
unzip $HOME/myprogs/source/libsvm-dense-3.17Modified.zip -d $HOME/myprogs/source

mv $HOME/myprogs/source/dlib-19.0 $HOME/myprogs/lib/dlib-19.0
mv $HOME/myprogs/source/libsvm-dense-3.17Modified $HOME/myprogs/lib/libsvm-dense-3.17Modified 

mkdir -p $HOME/myprogs/lib/libconfig

(cd $HOME/myprogs/source/libconfig-1.5/ ; ./configure --prefix=$HOME/myprogs/lib/libconfig)


make -C $HOME/myprogs/source/libconfig-1.5
make -C $HOME/myprogs/source/libconfig-1.5 install

#Execute these environment variable each time you want to execute the CPP Pipeline
export CPATH=$HOME/myprogs/lib/libsvm-dense-3.17Modified:$CPATH
export CPATH=$HOME/myprogs/lib/dlib-19.0:$CPATH
export CPATH=$HOME/myprogs/lib/libconfig/include:$CPATH
export LIBRARY_PATH=$HOME/myprogs/lib/libconfig/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/myprogs/lib/libconfig/lib:$LD_LIBRARY_PATH

#change into cpp pipeline directory
make -C $HOME/mycode/cppCode/exec all multiproc=1
