from ROOT import TH1F, TCanvas, kBlue, kRed
from data_simulation import ClusterSimulator

# draw the data simulation
print("")
print("------------------NEW--------------------------")
print("")
c2 = TCanvas()
simulator=ClusterSimulator("config1.json")
ngen = 5
c2 = TCanvas()
h = TH1F('h','',simulator.r,0,simulator.r)
for j in range(ngen):
    clus = simulator.generate_MIP_cluster()
    for i in range(len(clus)):
        h.SetBinContent(i+1,clus[i])
           
           
    h.SetLineColor(kRed)
    h.SetFillColorAlpha(kRed, 0.5) 
    h.SetFillStyle(3001) 
    
    h.Draw()
    c2.Update()
    c2.Modified()
    c2.Draw()
    c2.Print('prints/clus_'+str(simulator.r)+'_MIP_'+str(j)+'.png')
    #print(clus)
    print(sum(clus))
print("")
print("-----------------------------------------------")
print("")
c2 = TCanvas()
h = TH1F('h','',simulator.r,0,simulator.r)
for j in range(ngen):
    clus = simulator.generate_2MIP_cluster()
    for i in range(len(clus)):
        h.SetBinContent(i+1,clus[i])
           
           
    h.SetLineColor(kBlue)
    h.SetFillColorAlpha(kBlue, 0.5) 
    h.SetFillStyle(3001) 
    
    h.Draw()
    c2.Update()
    c2.Modified()
    c2.Draw()
    c2.Print('prints/clus_'+str(simulator.r)+'_2MIP_'+str(j)+'.png')
    #print(clus)
    print(sum(clus))

print("")
print("------------------END--------------------------")
print("")