## homework 2 [ report ](https://github.com/sliao7/CSE6740_Computational_Data_Analysis/blob/main/homework2/Shasha_Liao_HW2_report.pdf) 
### 1. Linear Dimension Reduction: PCA on food consumption in European countries [ code ](https://github.com/sliao7/CSE6740_Computational_Data_Analysis/blob/main/homework2/python/food_PCA.py)
* Studied the food consumption of 16 countries in Europe for 20 food items, such as tea, jam, and other.
* Derived and soved the mathematical optimization problem for finding top k principal components.
* Implemented the PCA algorithm from scrach.
* Visualized the data in a 2D plane using the two principal components.
### 2. Nonlinear Dimension Reduction: ISOMAP on order of faces [ code ](https://github.com/sliao7/CSE6740_Computational_Data_Analysis/blob/main/homework2/python/isomap.py)
* Reproduced the ISOMAP algorithm results in the original [paper](https://web.mit.edu/cocosci/Papers/sci_reprint.pdf) for nonlinear dimension reduction. [[1]](#1)
* Implemented the ISOMAP algorithm from scratch.
* Ran the ISOMAP algorithm on a dataset containing 698 images, corresponding to different poses of the same face.
* Visualized the result and compared it with the result from PCA.
![alt text][isomap_faces]

[isomap_faces]: https://github.com/sliao7/CSE6740_Computational_Data_Analysis/blob/main/homework2/Latex/isomap_face_scatter.png 

## References
<a id="1">[1]</a> 
 J.B. Tenenbaum, V. de Silva, and J.C. Langford.
 A Global Geometric Framework for Nonlinear Dimensionality Reduction.
 Science 290 (2000) 2319-2323.
