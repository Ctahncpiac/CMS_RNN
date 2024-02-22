from ROOT import TH1F, TCanvas, kBlue
from data_simulation import ClusterSimulator

# draw the data simulation
simulator=ClusterSimulator("config1.json")
ngen = 1
c2 = TCanvas()
h = TH1F('h','',simulator.r,0,simulator.r)
for j in range(ngen):
    clus = simulator.generate_MIP_cluster()
    for i in range(len(clus)):
        h.SetBinContent(i+1,clus[i])
       
    h.SetLineColor(kBlue)
    h.SetFillColorAlpha(kBlue, 0.5) 
    h.SetFillStyle(3001) 
    
    h.Draw()
    c2.Update()
    c2.Modified()
    c2.Draw()
    c2.Print('clus'+str(j)+'.png')
