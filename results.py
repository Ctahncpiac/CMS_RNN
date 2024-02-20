import ROOT
from ROOT import TMVA, TFile, TH1F, TCanvas, kRed, kBlue
from training import train
from simple_algorithm import SimpleAlgo
import matplotlib.pyplot as plt

def comparison():
    name = input("Please enter a file name: ")
    training=train()
    T=training.execute(3 , 100, 3, 1000, 0.8, 0.8,name,1) #use_type=3 , batchSize=100, maxepochs=50, nEvts=1000, pB=0.8, pS=0.8,name='test',method=1 by default
    use_type=training.use_type
    c1 = T[0].GetROCCurve(T[1])
    c1.Draw()
    c1.Print('ROC_'+name+'.png')


    rnn_types = ["TMVA_RNN", "TMVA_LSTM", "TMVA_GRU"]
    use_rnn_type = [1, 1, 1]
        
    if 0 <= use_type < 3:
        use_rnn_type = [0, 0, 0]
        use_rnn_type[use_type] = 1
    with open('AUC_'+name+'.txt', 'a') as f:       
        for i in range(3):
            if use_rnn_type[i]:
                print('AUC for the '+ rnn_types[i] + ' method: '+ str(T[0].GetROCIntegral(T[1],rnn_types[i])),file=f) 

    
    if T[2]:
        T[2].Close()  

def hist_hyp(type): #type = "integral" ; "charge" ; "width" ; "position"

        name = input("Please enter a file name: ")
        hyp = SimpleAlgo(name, 1000) #name , ngen
        s_sng, s_bkg = hyp.hyp_t(type)
        
        c2 = TCanvas()

        h1 = TH1F('h1', '',hyp.bins, hyp.vmin, hyp.vmax)
        h2 = TH1F('h2', '', hyp.bins, hyp.vmin, hyp.vmax)

        for j in s_sng:
            h1.Fill(j)
        for j in s_bkg:
            h2.Fill(j)

        h1.SetLineColor(kRed)
        h1.SetFillColorAlpha(kRed, 0.5) 
        h1.SetFillStyle(3004) 

        h2.SetLineColor(kBlue)
        h2.SetFillColorAlpha(kBlue, 0.5) 
        h2.SetFillStyle(3005) 

        h1.Draw()
        h2.Draw('same') 

        c2.Draw()
        c2.Print("Hyp_" +type +'_'+ name + '.png')

def plot_roc_curve(type): #type = "integral" ; "charge" ; "width"
        # Plot the ROC curve
        name = input("Please enter a file name: ")
        algo = SimpleAlgo(name,1000)
        sng_e, bkg_r = algo.evaluate_performance(type,2000)  #threshold
        plt.figure()
        plt.plot(sng_e,bkg_r, color='darkorange', label='ROC curve')
        plt.plot([0, 1], [1, 0], color='navy', lw=0.5, linestyle='--')
        plt.ylabel('Background rejection')
        plt.xlabel('Signal efficiency')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

#plot_roc_curve("position")
hist_hyp("charge")






