# CNN implementation on C++
* 大助教: 吳郁夫
* Email: yves.eo98g@nctu.edu.tw
* LineID: yveswu
* 出席：10%
* 作業：50%
* 期中報告：10% (deadline 11/8 23:59 in pdf)
* 期末專題：30%
## Part 1. Implement LeNet in C++
#### Introduction
* explore the structure of LeNet
* learning method and implement Neural Network on C++
#### Reference and Needs
* [A ground-up C++ convnet that scores 0.973 on the Kaggle Digit Recognizer Challenge](https://plantsandbuildings.github.io/machine-learning/misc/math/2018/04/28/a-ground-up-c++-convnet-that-scores-0.973-on-the-kaggle-digit-recognizer-challenge.html)
* Linux OS / Virtual Box on Windows / Windows subsystem for linux 2
* [Armadillo](http://arma.sourceforge.net/docs.html) >= 8.300.4
    * doing matrix maths
    * high quality linear algebra library
    * similar to Matlab
    * we only use in matrix computation
    ```
    #include <armadillo>
    using namespace arma;
    int main() {
       ... 
    }
    ```
    ```
    #include <armadillo>
    int main() {
        arma::vec b;
        arma::mat A;
        ...
    }
    ```

* Boost unit test framework >= 1.58
    * Testing tool
    ```
    #include <boost/test/unit_test.hpp>
    BOOST_AUTO_TEST_CASE(my_test) {
        BOOST_CHECK(add(2,2)==4);
        ...
    }
    ```
#### Evolution of ANN
* computer needs more power and ram to solve a problem that human can easily figure out
* let computer thinks like human (recognition)
![figure 1](https://i.imgur.com/pJwvUSs.png)
* Human's Neurals &rarr; Perceptron &rarr; Multilayer Perceptron &rarr; Deep Neural Network
#### LeNet-5 Architecture
![LeNet5](https://i.imgur.com/smsV6dn.jpg)
* convolution 的目的是要找到細微特徵（特徵提取）
* 1st layer(2D convolution)
    * input: image
    * output: 6 feature maps
    * find out features
    ![C1](https://i.imgur.com/2V7jXn3.png)
    * Conv2D
        * Kernel 5*5 (?)
        * Kernel will be used reapeatly
        ![conv2D](https://s3-us-west-2.amazonaws.com/static.pyimagesearch.com/keras-conv2d/keras_conv2d_padding.gif)
* 2nd layer
    * down sampling
    * some implementation doesn't have this
    ![](https://i.imgur.com/oBPLyrN.png)
    * maximum pooling
    ![](https://i.imgur.com/hL2tAQU.png)
    ![](https://i.imgur.com/ax4O7w1.png)
* 3rd layer
    * input: 6 images
    * output: 16 feature maps
    ![](https://i.imgur.com/NeNzicn.png)
    ![](https://i.imgur.com/B0F2CRe.png)
    * Conv2D on Multiple Frames
    ![Conv2D_mul](https://i.imgur.com/po8Sqw7.png)
    * 2D and 3D convolution
    ![](https://i.imgur.com/hKHJOl4.jpg)
* 4th layer
    * down sampling
    ![](https://i.imgur.com/5DOYxEN.png)

* 5th and 6th layer
    * 5th
    ![](https://i.imgur.com/7DmlNOH.png)
    * 6th
    ![](https://i.imgur.com/Gv0RNh1.png)
    * output
    ![](https://i.imgur.com/nUMtU85.png)
* summary
    ![](https://i.imgur.com/2Lj9Fsf.jpg)
    * 實作前先計算Neural Network中的參數數量
#### Kaggle dataset
![](https://i.imgur.com/zammHq2.jpg)
#### Implemention
* file structure
    ```
    - LeNet
        - LeNet.cpp
        - layers
            - convolution_layer.hpp
            - cross_entropy_loss_layer.hpp
            - dense_layer.hpp
            - max_pooling_layer.hpp
            - relu_layer.hpp
            - softmax_layer.hpp
    ```
* Le_net.cpp
    ```
    #include "layers/xxx_layer.hpp"
    #define DEBUG true
    int main() {
        // Read the Kaggle data
        // Define the network layers
        // 訓練次數
        for (size_t epoch=0; epoch<EPOCH; epoch++) {
            // Generate a random batch
            // 分批計算(減少參數的更新次數)
            for (size_t i=0; i<BATCH_SIZE; i++) {
                // Forward pass
                // Compute the loss
                // Backward pass
            }
            // Update params
        }
        // Compute the training accuracy
        // Compute validation accuracy
        // Reset cumulative loss and correct count
        // Write results on test data to result csv
    }
    ```
* convolution_layer.hpp
    ```
    class ConvolutinoLayer {
        public:
         ConvolutinoLayer(...):...{ //init }
         void Forward(...) { // forward pass}
         void Backward(...) { //backward pass}
         void UpdateFilterWeights(...) {...}
         ...
    }
    ```
* forward and backward propagatino
![for_back](https://miro.medium.com/max/1170/1*0hf4gLbc-2V5RMXBhluJ_A.gif)
* [ReLU Layer](https://mropengate.blogspot.com/2017/02/deep-learning-role-of-activation.html)
    * 計算量小，只需判斷輸入是否大於零，不用指數運算
    ![](https://i.imgur.com/QTKfYv0.png)
    ![](https://i.imgur.com/IcXCOg6.png)
    * 為何不用 TanH 而選擇用 ReLU?

## Homework1
* 正確率
* 花費時間
* 最好結果的 csv 檔
* 嘗試修改 activation function -> tanH, fully connect 比較差異
* 嘗試將正確率超越 0.973 (Bonus!)
#### Homework on Mac
```
> git clone https://github.com/PinmanHuang/cpp-cnn.git
> cd cpp-cnn; mkdir build data
Then clone Kaggle Digit Recognizer dataset into the data directory.
```

```
> cd build
> brew install cmake
> cmake ../
You might have cMake Error: Could NOT find Boost
```
```
Solution
> brew install boost-python
> cmake ../
You might have cMake Error:Could NOT find Armadillo
```
```
Solution
> brew install armadillo
> cmake ../
Finally, it configures successfully and generates the Makefile. And then buiding the project.
> make
```
```
Training the data.
> ./bin/le_net
```
#### Homework on Linux Ubuntu
```
> git clone https://github.com/PinmanHuang/cpp-cnn.git
> cd cpp-cnn; mkdir build data
Then clone Kaggle Digit Recognizer dataset into the data directory.
```
```
Install the cMake.
> sudo apt-get install cmake
> cd build
> cmake ../
You might have cMake Error: Unable to find the requested Boost libraries.
```
```
Solution
> apt-get install libboost-all-dev
> cmake ../
You might have cMake Error:Could NOT find Armadillo.
```
```
Solution
> apt-get install libopenblas-dev liblapack-dev wget
> wget http://sourceforge.net/projects/arma/files/armadillo-9.700.2.tar.xz
> tar -xvf armadillo-9.700.2
> cd armadillo-9.700.2
> ./configure
> make
> make install
> cmake ../
Finally, it configures successfully and generates the Makefile. And then buiding the project.
> make
```
```
Training the data.
> ./bin/le_net
```
#### Homework on Windows
```
Installing cygwin first.
Packages that you should install:
gcc, gcc-g++, cmake, make, boost, wget, libopenblas, liblapack
```
```
> git clone https://github.com/PinmanHuang/cpp-cnn.git
> cd cpp-cnn; mkdir build data
Then clone Kaggle Digit Recognizer dataset into the data directory.
```
```
> wget http://sourceforge.net/projects/arma/files/armadillo-9.700.2.tar.xz
> tar -xvf armadillo-9.700.2.tar.xz
> cd armadillo-9.700.2
> ./configure
> make
> make install
> cmake ../
```
```
Fail!!!!!!!!!
```