Neural network to recognize handwritten digits

Data set: http://archive.ics.uci.edu/ml/datasets/Semeion+Handwritten+Digit

Program uses sample set of 400 digits (40 of each digit) from Semeion handwritten digits set. Set is divided into two halfs of training and testing sets. I have modified the data set by removing decimals for easier data parsing.

Example values for testing: 2 layers, 100 nodes, 500 epochs

* More epochs and less nodes seems to be better than the other way around
* High number of layers destroys the results

Program doesn't run after building?

Make sure to include "semeion_sample.data" in your build directory.

![ann1](https://user-images.githubusercontent.com/28627738/73890567-f3b29b00-487a-11ea-9a47-6adf38680ea3.png)

![ann2](https://user-images.githubusercontent.com/28627738/73890677-307e9200-487b-11ea-98d7-2eb7eea04e8d.png)
