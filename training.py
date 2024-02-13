import ROOT
from RNN import Classification
from ROOT import TMVA, TFile

C=Classification(name='test',ngen=10000, writeOutputFile=True)

factory=C.tmva(type=3 , batchSize=100, maxepochs=10, nEvts=1000, pB=0.8, pS=0.8)
#factory=C.PyMVA(type=3 , batchSize=100, maxepochs=10, nEvts=1000, pB=0.8, pS=0.8)

## Train all methods
factory.TrainAllMethods()
print("nthreads  = {}".format(ROOT.GetThreadPoolSize()))
        
# ---- Evaluate all MVAs using the set of test events
factory.TestAllMethods()
        
        # ----- Evaluate and compare performance of all configured MVAs
factory.EvaluateAllMethods()
 
