# IDBSCAN
This is an implementation of IDBSAN which stands for Intersection DBSCAN. Before applying DBSCAN we locate the leaders with the improved leader* algorithm. 
The main steps of IDBSCAN full applications are the following:
1. Apply Leader* to find the leaders and their corresponding followers, while allowing more than one leader to each example.
2. Apply a sampling of the intersected samples so that a dense leader in the original data will remain dense in the created sub-data. 
3. Apply DBSCAN on the sub-data (S_data) that contains both the leaders and the sampled examples from step 2. 
4. Get the prediction of the *leaders* and pass their predictions to their followers.  
