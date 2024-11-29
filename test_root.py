import ROOT

def myFunc(x, par):
    return par[0]*x[0] + par[1]


test_fun = ROOT.TF1("myFunc",myFunc,0,100,2)
test_fun.SetParameter(0,1)
test_fun.SetParameter(1,0)

c1 = ROOT.TCanvas('c1','myCanvas',10,10,800,800)
test_fun.Draw()