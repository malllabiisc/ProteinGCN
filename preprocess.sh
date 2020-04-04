git clone https://github.com/gjoni/mylddt
mv mylddt ./preprocess
cd preprocess/src/
g++ -Wall -Wno-unused-result -pedantic -O3 -mtune=native -std=c++11 *.cpp -o get_features
mv get_features ../
