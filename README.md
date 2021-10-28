# I-DBSCAN
This is an implementation of IDBSAN which stands for Intersection DBSCAN. 

### DBSCAN

Density-Based Spatial Clustering of Applications with Noise is a classic clustering algorithm that has a major advantage of capturing amorphic clusters, where other clustering algorithms, such as k-means, fails. 




![DBSCAN-algorithm](https://github.com/tairtahar/IDBSCAN/blob/master/images/DBSCAN_pseudo.PNG) ![DBSCAN-clustering](https://github.com/tairtahar/IDBSCAN/blob/master/images/dbscan_clusters.PNG)


### I-DBSCAN

Before applying DBSCAN we locate the leaders with the improved leader* algorithm. The goal of this approach is to reduce DBSCAN algorithm complexity by running it on reduced set of samples that are the representatives of the whole dataset. 
The main steps of IDBSCAN full applications are the following:

1. Apply Leader* to find the leaders and their corresponding followers, while allowing more than one leader to each example.
2. Apply a sampling of the intersected samples so that a dense leader in the original data will remain dense in the created sub-data. 
3. Apply DBSCAN on the sub-data (S_data) that contains both the leaders and the sampled examples from step 2. 
4. Get the prediction of the *leaders* and pass their predictions to their followers.  


To execute the code go to main.py and adjust the parameters as you wish:
1. Choose number for dataset to use from the following possible: "abalone" - 0, "mushroom" - 1, "pendigit" - 2, "letter" - 3, "cadata" - 4, "sensorless" - 5, "shuttle" - 6.
2. Which algorithms you would wish to execute. It is possible to execute all of them at once: "IDBSCAN", "DBSCAN", "stdbscan", "hdbscan", "leader". 
3. flag_save if you wish to save the clustering of IDBSCAN to txt file.
4. path - in case of flag_save == True.
5. verbose - True if you are interested in seeing more detailed results/tracking the execution details.

####Please approach the "Report.pdf" file for deeper explanations on each of the algorithms, their implementations and eventually Experiments results. 

### Citing

Luchi, Diego, Alexandre L. Rodrigues and Flávio Miguel Varejão. “Sampling approaches for applying DBSCAN to large datasets.” *Pattern Recognit. Lett.* 117 (2019): 90-96.

> @article{Luchi2019SamplingAF,
>   title={Sampling approaches for applying DBSCAN to large datasets},
>   author={Diego Luchi and Alexandre L. Rodrigues and Fl{\'a}vio Miguel Varej{\~a}o},
>   journal={Pattern Recognit. Lett.},
>   year={2019},
>   volume={117},
>   pages={90-96}
> }