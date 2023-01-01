# Linear Regression Using Binary Search 🔍📉

When I first learned linear regression, and found that it involves an [optimization](https://en.wikipedia.org/wiki/Mathematical_optimization) of a [convex function](https://en.wikipedia.org/wiki/Convex_function), I had the idea that it could also be optimized using binary search.

Linear regression involves finding the optimal coefficients for a linear model, and it's generally either solved analytically or using gradient descent (which one is used depends on the size of the dataset and the number of features being fitted on).

To my delight, binary search actually worked! you can experiment with it yourself (the code is very simple).