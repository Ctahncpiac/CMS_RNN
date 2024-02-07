from ROOT import TH1F, TCanvas
from data_simulation import ClusterSimulator

# draw the data simulation
ngen = 1000
c2 = TCanvas()
h = TH1F('h','',10,0,10)
for i in range(ngen):
    simulator=ClusterSimulator()
    clus = simulator.generate_MIP_cluster()
    for i in range(10):
        h.SetBinContent(i,clus[i])
    h.Draw()
    c2.Update()
    c2.Modified()
    c2.Draw()
