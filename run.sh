cd build
cmake ..
make
cd ..

CUDA_VISIBLE_DEVICES=3 build/main -k 1