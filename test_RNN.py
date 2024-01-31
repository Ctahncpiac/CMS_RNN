#!/usr/bin/env python
# coding: utf-8

# 
# Use of TMVA
# 
# Simulation of several categories of clusters:
#   - MIP 
#   - double MIP
#   - flat
#   
#   
# 
# 

# Setup:
#   - ROOT
#   

# In[1]:

import ROOT 
from ROOT import TTree, TRandom3


# Create function to simulate clusters
# 

# In[2]:


rand = TRandom3()
from array import array

#G stands for Generate

#Generate a position between 0 and 1 within a strip
def GPosition():
    return rand.Uniform(1)
    
#Generate the mean charge (mean of landau)    
def GMeanQ(mu,sig):
    return rand.Gaus(mu,sig)
    
#Generate the total charge    
def GQ(mu,sig):
    return rand.Landau(mu,sig)
    
def GMIPCluster(q = 100, sigG = 10, sigL = 20, xt0 = 0.8, xt1=0.1, noise = 8):
    pos = GPosition()
    Q = GQ(GMeanQ(q,sigG),sigL)
    #create an array of size 10
    clus = [0]*10
    #charge the main charge among 2 clusters based on the input position
    #print(Q,pos)
    clus[4] = 12
    clus[4] = Q*pos
    clus[5] = Q*(1-pos)
    #apply a cross-talk effet
    tmp_3 = clus[4]*xt1
    tmp_4 = clus[4]*xt0+clus[5]*xt1
    tmp_5 = clus[5]*xt0+clus[4]*xt1
    tmp_6 = clus[5]*xt1
    clus[3] = tmp_3
    clus[4] = tmp_4
    clus[5] = tmp_5
    clus[6] = tmp_6
    #add noise
    for i in range(len(clus)):
       clus[i]+=rand.Gaus(0,noise)
    #apply threshold
    for i in range(len(clus)):
        if clus[i]<3*noise: clus[i]=0
    return clus


def G2MIPCluster(delta_pos=3,q = 100, sigG = 10, sigL = 20, xt0 = 0.8, xt1=0.1, noise = 8):
  clus1 = GMIPCluster(q,sigG,sigL,xt0,xt1,noise)
  clus2 = GMIPCluster(q,sigG,sigL,xt0,xt1,noise)
  dp = round(rand.Uniform(1,delta_pos)) 
  #print(dp)
  clusd = [0]*10
  for i in range(len(clusd)):
        if (i+dp)<len(clusd): clusd[i] = clus1[i] + clus2[i+dp]
        else: clusd[i] = clus1[i]
        #print(i,clus1[i],clus2[i],clusd[i])
  return clusd

print(G2MIPCluster())
        
    
    
    
    


# Test the generation by generating distribution on histo
# 
# 

# In[3]:


from ROOT import TH1F, TCanvas

ngen = 1000
c2 = TCanvas()
h = TH1F('h','',10,0,10)
for i in range(ngen):
    clus = GMIPCluster()
    for i in range(10):
        h.SetBinContent(i,clus[i])
    h.Draw()
    c2.Update()
    c2.Modified()
    c2.Draw()

    


# In[ ]:





# In[4]:


from ROOT import TFile, TTree

f = TFile('tree.root', "RECREATE")
tsgn = TTree("tsgn", "sgn")
tbkg = TTree("tbkg", "bkg")

clusSize = 10
xsng = array('f',[0]*clusSize)
xbkg = array('f',[0]*clusSize)
#for i in range(clusSize):
#    tbkg.Branch("clus_" + str(i), xbkg[i], "clus_" + str(i) + "/F")
#    tsgn.Branch("clus_" + str(i), xsng[i], "clus_" + str(i) + "/F")
tbkg.Branch("clus", xbkg, "xbkg[10]/F")
tsgn.Branch("clus", xsng, "xsng[10]/F")    

ngen = 10000
for i in range(ngen):
    onemip = GMIPCluster()
    for i in range(len(onemip)):
        xsng[i] = onemip[i]
    twomip = G2MIPCluster()
    for i in range(len(onemip)):
        xbkg[i] = twomip[i]
    tsgn.Fill()
    tbkg.Fill()
f.Write()
f.Print()
f.Close()


# Test the file

# In[5]:


#f = TFile("tree.root","READ")
#tsng = f.Get("tsgn")
#c = TCanvas()
#tsng.Draw("clus[4]","clus[4]<100 && clus[4]>1")
#c.Draw()


# Instructions
# 

# In[6]:


from ROOT import TMVA
import ROOT

archString = "CPU" #could be "GPU"
rnn_types = ["RNN", "LSTM", "GRU"]
use_rnn_type = [1, 1, 1]
use_type = 1
batchSize = 100
maxepochs = 10

outputFile = TFile.Open("data_RNN.root", "RECREATE")

# // Creating the factory object
factory = TMVA.Factory(
    "TMVAClassification",
    outputFile,
    V=False,
    Silent=False,
    Color=True,
    DrawProgressBar=True,
    Transformations=None,
    Correlations=False,
    AnalysisType="Classification",
    ModelPersistence=True,
)
dataloader = TMVA.DataLoader("dataset")
inputFile  = TFile("tree.root","READ")
signalTree = inputFile.Get("tsgn")
background = inputFile.Get("tbkg")

ninputs = 10
dataloader.AddVariablesArray("clus", ninputs)

# check given input
datainfo = dataloader.GetDataSetInfo()
vars = datainfo.GetListOfVariables()
print("number of variables is {}".format(vars.size()))
 
 
for v in vars:
    print(v)
    
    
nTotEvts = 10000    
nTrainSig = 0.5 * nTotEvts  #Default : 0.8
nTrainBkg = 0.7 * nTotEvts  #Default : 0.8

# Apply additional cuts on the signal and background samples (can be different)
mycuts = ""  # for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
mycutb = ""

# build the string options for DataLoader::PrepareTrainingAndTestTree
dataloader.PrepareTrainingAndTestTree(
    mycuts,
    mycutb,
    nTrain_Signal=nTrainSig,
    nTrain_Background=nTrainBkg,
    SplitMode="Random",
    SplitSeed=100,
    NormMode="NumEvents",
    V=False,
    CalcCorrelations=False,
)

print("singalTree",type(signalTree))
dataloader.AddSignalTree(signalTree, 1.0)
dataloader.AddBackgroundTree(background, 1.0)


for i in range(3):
        if not use_rnn_type[i]:
            continue
 
        rnn_type = rnn_types[i]
 
        ## Define RNN layer layout
        ##  it should be   LayerType (RNN or LSTM or GRU) |  number of units | number of inputs | time steps | remember output (typically no=0 | return full sequence
        rnnLayout = str(rnn_type) + "|8|" + "1" + "|" + str(ninputs) + "|0|1,RESHAPE|FLAT,DENSE|64|TANH,LINEAR"
 
        ## Defining Training strategies. Different training strings can be concatenate. Use however only one
        trainingString1 = "LearningRate=1e-3,Momentum=0.0,Repetitions=1,ConvergenceSteps=5,BatchSize=" + str(batchSize)
        trainingString1 += ",TestRepetitions=1,WeightDecay=1e-2,Regularization=None,MaxEpochs=" + str(maxepochs)
        trainingString1 += "Optimizer=ADAM,DropConfig=0.0+0.+0.+0."
 
        ## define the inputlayout string for RNN
        ## the input data should be organize as   following:
        ##/ input layout for RNN:    time x ndim
        ## add after RNN a reshape layer (needed top flatten the output) and a dense layer with 64 units and a last one
        ## Note the last layer is linear because  when using Crossentropy a Sigmoid is applied already
        ## Define the full RNN Noption string adding the final options for all network
        rnnName = "TMVA_" + str(rnn_type)
        factory.BookMethod(
            dataloader,
            TMVA.Types.kDL,
            rnnName,
            H=False,
            V=True,
            ErrorStrategy="CROSSENTROPY",
            VarTransform=None,
            WeightInitialization="XAVIERUNIFORM",
            ValidationSize=0.2,
            RandomSeed=1234,
            InputLayout=str(ninputs) + "|" + "1",
            Layout=rnnLayout,
            TrainingStrategy=trainingString1,
            Architecture=archString
        )
        
## Train all methods
factory.TrainAllMethods()
 
print("nthreads  = {}".format(ROOT.GetThreadPoolSize()))
 
# ---- Evaluate all MVAs using the set of test events
factory.TestAllMethods()
 
# ----- Evaluate and compare performance of all configured MVAs
factory.EvaluateAllMethods()
 
# check method
 
# plot ROC curve
c1 = factory.GetROCCurve(dataloader)
c1.Draw()
c1.Print('2.png')
 
if outputFile:
    outputFile.Close()  


# In[8]:


# After running the MVA code, a file called TMVAClassification_<MVAmethodname>.weights.xml,
# where <MVAmethodname> is the name of the algorithm you used, for instance BDT,
# is created in the folder dataset/weights
# This file contains the results of the training and it's used to apply the MVA to our data
# As for the training, each event on data will have a MVA value assigned, that will tell us how "signal-like" that event is
# No worries, the procedure is automatic and you only need a few commands!

#This code works for a BDT. If you want to use another method you need to change
#all the istances of "BDT" with the name of your MVA algorithm.

# we need to use the 'array' module. This is a technical detail that you can ignore
from array import array
from ROOT import TMVA,TFile


f = ROOT.TFile.Open("tree_toberead.root")
t = f.Get("t")

# let's initialize the TMVA Reader that is used to get the BDT value
reader = TMVA.Reader("!Silent")


fmva = ROOT.TFile.Open("data_RNN.root")
tmva=fmva.Get("dataset/TrainTree")
# the following lines take the variables used in the training and use their values to compute the BDT
List=["clus"]
br = {}
clus0 = 0
clus1 = 0
clus2 = 0
clus3 = 0
clus4 = 0
clus5 = 0
clus6 = 0
clus7 = 0
clus8 = 0
clus9 = 0

for n in range(10):
    exec('clus'+str(n)+' = array(\'f\',[0])')
    print('clus'+str(n)+' = array(\'f\',[0])')
    exec('reader.AddVariable("clus",clus'+str(n)+')')
    #exec('reader.AddVariable("clus'+str(n)+'",clus'+str(n)+')')
    print('reader.AddVariable("clus",clus'+str(n)+')')
    #exec('Vars.append(clus'+str(n)+')')

 

"""
for branch in tmva.GetListOfBranches():
  branchName = branch.GetName()
  print(branchName)
  if branchName  in List:
    clus = array('f', [-999.]*10)
    reader.AddVariable("clus", clus0)
    reader.AddVariable("clus", clus1)
    reader.AddVariable("clus", clus2)
    reader.AddVariable("clus", clus3)
    reader.AddVariable("clus", clus4)
    reader.AddVariable("clus", clus5)
    reader.AddVariable("clus", clus6)
    reader.AddVariable("clus", clus7)
    reader.AddVariable("clus", clus8)
    reader.AddVariable("clus", clus9)
    #lclus = [array('f',clus[i]) for i in range(10)]
    #for i in range(10):
           #reader.AddVariable("clus["+str(i)+"]", lclus[i])
           #reader.AddVariable("clus", clus[i])
            #dd"br[branchName]["+str(i)+"]")
"""

#for(Long64_t i=0; i < vector_size; ++i) {
#   TString branch_name = Form("B_px[%i]", i);
#   dataloader->AddVariable(branch_name, branch_name, "units", 'F' );
#}
#    reader.AddVariable('floatb', br[branchName][i],'floatb')
    #t.SetBranchAddress(branchName, clus)

# let's tell the reader where the training is stored
reader.BookMVA("TMVA_GRU", "dataset/weights/TMVAClassification_TMVA_GRU.weights.xml")

#Create the new file and the new TTree
#data_fit = "/content/drive/MyDrive/data_for_fit.root" if RunningInCOLAB else "data_for_fit.root"
#newFile = ROOT.TFile.Open("tree_toberead.root","RECREATE")
#newTree = newFile.Get("t")


MVA = array('f',[0])



n = t.GetEntries()
print(type(clus))

# let's now read the events in the data and save only the events that pass the cut on the BDT
for i in range(n):
    t.GetEntry(i)
    clus = t.clus
   
    #print(lclus)
    """
    clus0 = clus[0]
    clus1 = clus[1]
    clus2 = clus[2]
    clus3 = clus[3]
    clus4 = clus[4]
    clus5 = clus[5]
    clus6 = clus[6]
    clus7 = clus[7]
    clus8 = clus[8]
    clus9 = clus[9]
    """
    for i in range(10):
       exec('clus'+str(i)+' = t.clus['+str(i)+"]")
    
    #print(i,t.clus[4], clus4, clus)
    MVA[0] = reader.EvaluateMVA(t.clus,"TMVA_GRU") #here the MVA is evaluated for the event i
    #MVA[0] = reader.EvaluateMVA(t.clus) #here the MVA is evaluated for the event i

    print(MVA[0])
    #newmass[0] = mass[0]
    #newtag[0] = tag[0]
    #newTree.Fill()
    #print("MVA = " + str(MVA[0]))
    #print("mass e tag = " + str(mass[0]) + " " + str(tag[0]))


#Tree.Write("",TFile.kOverwrite)
#newFile.Close()


# In[ ]:


f = ROOT.TFile.Open("tree_toberead.root")
t = f.Get("t")
for i in range(t.GetEntries()):
    t.GetEntry(i)
    print(t.clus[4])
    print(type(t.clus))


# In[ ]:


from ROOT import TH1F, TLegend, TCanvas
# let's initialize the TMVA Reader that is used to get the BDT value
reader = TMVA.Reader("!Silent")

for n in range(10):
    exec('clus'+str(n)+' = array(\'f\',[0])')
    print('clus'+str(n)+' = array(\'f\',[0])')
    exec('reader.AddVariable("clus",clus'+str(n)+')')
    #exec('reader.AddVariable("clus'+str(n)+'",clus'+str(n)+')')
    print('reader.AddVariable("clus",clus'+str(n)+')')
    #exec('Vars.append(clus'+str(n)+')')

# let's tell the reader where the training is stored
reader.BookMVA("TMVA_GRU", "dataset/weights/TMVAClassification_TMVA_GRU.weights.xml")
    
qs = [50,75,90,100,110,125,150,200]
histo = {q:TH1F("h_"+str(q),"",100,0,1) for q in qs}
eff = {q:0 for q in qs}
cut = 0.8
ngen=10000
color=1
for h in histo.values():
    h.SetLineColor(color)
    color+=1
first = True
c = TCanvas()
leg = TLegend()
for q in qs:
   for i in range(ngen):
      cluster = GMIPCluster(q)
      mva = reader.EvaluateMVA(cluster,"TMVA_GRU")
      histo[q].Fill(mva)
      if mva>cut: 
        eff[q]+=1
   leg.AddEntry(histo[q],"q = "+str(q))
   if first: 
      histo[q].Draw()
      first = False
   else: histo[q].Draw("same")
   eff[q]/=ngen
   print("q = ",q,"eff > 0.9 = ",eff[q])
leg.Draw()
c.Draw()
c.Print('3.png')


# In[ ]:




