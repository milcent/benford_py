# Benford for Python
[Benford's Law](https://en.wikipedia.org/wiki/Benford%27s_law) is the unintuitive fenomenom beared by certain series of data that makes the proportions of the first digits of such numbers be higher for the lower ones (1, 2, 3,...) than the proportions of the higher ones (8, 9).

The expected distributions of the First Digits in a Benford-compliant data set are the ones shown below:
![First Digits](https://github.com/milcent/benford_py/blob/master/img/First.png)

Benford's Law has been used in [several fields](http://www.benfordonline.net/). Afer asserting that the usual data type is Benford-compliant, one can study samples from the same data in search of inconsistencies, errors or even [fraud](https://www.amazon.com.br/Benfords-Law-Applications-Accounting-Detection/dp/1118152859).

This open source module is an attempt to facilitate the performance of Benford's Law-related tests by people using Python, whether interactively or in an automated, scripting way.

It uses the versatility of numpy and pandas, along with matplotlib for vizualization, to deliver results like the one bellow and much more.

![Sample Image](https://github.com/milcent/benford_py/blob/master/img/SPY-f2d-conf_level-95.png)

### Installation

cd into the site-packages subfolder of your python distribution and git clone from there:

```
 git clone https://github.com/milcent/benford.git
```
For a quick start, please go to the Demo notebook, in which I show examples on how to run the tests with the SPY (S&P 500 ETF) daily returns.

I will be adding information about the tests already available and also documentation.

I've been testing it in Python 2 and 3, and in Linux (Ubuntu), Windows and Mac, so feel free to file a bug report if you run into some trouble.

Also, if you have some nice data set that we can run these tests on, send it over an I will dedicate a jupyter notebook to it.

Thanks!

Milcent
