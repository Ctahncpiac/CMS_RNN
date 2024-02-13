import ROOT
from RNN import Classification
from ROOT import TMVA, TFile

C=Classification()
factory=C.tmva()
## Train all methods
factory.TrainAllMethods()
print("nthreads  = {}".format(ROOT.GetThreadPoolSize()))
        
# ---- Evaluate all MVAs using the set of test events
factory.TestAllMethods()
        
        # ----- Evaluate and compare performance of all configured MVAs
factory.EvaluateAllMethods()
 
