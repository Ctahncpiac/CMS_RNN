import ROOT
from RNN import Classification
from ROOT import TMVA, TFile


class train:
        def execute(self, use_type, batchSize, maxepochs, nEvts, pB, pS, name ,method):
                
                self.newname=name
                self.use_type=use_type
                self.batchSize=batchSize
                self.maxepochs=maxepochs
                self.nEvts=nEvts
                self.pB=pB
                self.pS=pS
                self.method=method

                C=Classification(self.newname,10000)
                if self.method==1:
                        factory=C.tmva(self.use_type, self.batchSize, self.maxepochs, self.nEvts, self.pB, self.pS)
                else:
                        factory=C.PyMVA(self.use_type, self.batchSize, self.maxepochs, self.nEvts, self.pB, self.pS)

                ## Train all methods
                factory[0].TrainAllMethods()
                print("nthreads  = {}".format(ROOT.GetThreadPoolSize()))
                        
                # ---- Evaluate all MVAs using the set of test events
                factory[0].TestAllMethods()
                        
                # ----- Evaluate and compare performance of all configured MVAs
                factory[0].EvaluateAllMethods()
                return factory
                


