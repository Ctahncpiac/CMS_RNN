# CMS_RNN
Recurrent Neural Network for CMS

##################################
###        DEPENDENCIES        ###
##################################

> python
> ROOT

##################################
###        ARCHITECTURE        ###
##################################

config1.json
|
|_ data_simulation.py
        |
        |_data_file.py 
        |         |
        |         |_simple_algorithm.py  ___________________________
        |         |                                                |
        |         |_RNN.py                                         |_results.py       
        |             |                                            |    
        |             |_training.py _______________________________|
        |
        |_draw.py  

##################################
###        How to use it       ###
##################################

First of all, you need to go to a directory containing the json file, as well as the clusters, data_root and 
prints directories, and all the py files. Beforehand, you'll need to install python, ROOT and the packages used in 
the various files (time, numpy, random, json, array, matplotlib.pyplot, tensorflow, keras). 
Then you can compile the code in the order of the architecture.
