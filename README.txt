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

|
|_ data_simulation.py
        |
        |_data_file.py__{TFile, TTree }  
                |
                |_simple_algorithm.py  ___________________________
                |                                                |
                |_RNN.py ___{ TMVA methods & Keras }             |_results.py  #mostly graphics#       
                    |                                            |    
                    |_training.py _______________________________|

  
