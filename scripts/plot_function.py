import sys,math,ctypes,array
import ROOT
import matplotlib.pyplot as plt
from ROOT import gROOT, gPad, gStyle

def qmuA(s,b): #Asimov q_mu, assuming mu_hat = 0 and mu = 1
  return 2*(-b*math.log(1+s/b)+s) 

def sigma(hs,hb):
  if hs.GetNbinsX()!=hb.GetNbinsX() :
    return 0
  else:
    sum_qmuA = 0 #Sum of -2ln(L(mu)/L(mu_hat))
    for ibin in range(1,hs.GetNbinsX()+1):
      if hb.GetBinContent(ibin) > 0:
        sum_qmuA += qmuA(hs.GetBinContent(ibin),hb.GetBinContent(ibin))
    #print("sum_qumA: ",sum_qmuA)
    return 1/math.sqrt(sum_qmuA)

def limit(hs,hb):
  sig = sigma(hs,hb)
  #print("sigma: ",sig)
  alpha = 0.05 #95% CLs
  return sig*ROOT.Math.gaussian_quantile(1-alpha*0.5,1)

def qmu_float(hs,hb,mu):
  if hs.GetNbinsX()!=hb.GetNbinsX() :
    return 0
  else:
    sum_qmu = 0 #Sum of -2ln(L(mu)/L(mu_hat))
    for ibin in range(1,hs.GetNbinsX()+1):
      if hb.GetBinContent(ibin) > 0:
        s_i = hs.GetBinContent(ibin)
        b_i = hb.GetBinContent(ibin)
        print ("signal is ", s_i)
        print ("BKG is ", b_i)
        sum_qmu += 2*(-b_i*math.log(1+s_i*mu/b_i)+s_i*mu)
    return sum_qmu

def max_value(value1, value2):
    if value1>=value2:
       return value1
    else:
       return value2 

def Read_Hist_Directory(file,samplename,categoryname,dir_list):
   for key_sample in file.GetListOfKeys():
    if samplename in key_sample.GetName():
      obj_sample = key_sample.ReadObj()
      if isinstance(obj_sample, ROOT.TDirectoryFile):
        for key_category in obj_sample.GetListOfKeys():
          if categoryname in key_category.GetName():
            obj_category = key_category.ReadObj()
            if isinstance(obj_category, ROOT.TDirectoryFile):
              dir_list.append(obj_category)

def Read_Hist(dir_list,samplename):
   for i, directory in enumerate(dir_list):
    directory.cd()
    hist_list = []
    for key in directory.GetListOfKeys():
      obj = key.ReadObj()
      if isinstance(obj, ROOT.TH3):
        hist_list.append(obj)

    for j, hist in enumerate(hist_list):
      if samplename == "QCD":
        hist.SetLineColor(3)
        hist.SetFillColor(3)
      if samplename == "TT":
        hist.SetLineColor(4)
        hist.SetFillColor(4)
      if samplename == "signal":
        hist.SetLineColor(2)
      if j == 0 and i == 0:
        hist_total = hist.Clone("hist_total")
      else:
        hist_total.Add(hist)
   return hist_total


def plot_bkg(cut,variable,category,bkg_list):
  hStack = ROOT.THStack()
  canvas = ROOT.TCanvas()
  canvas.SetLogy()
  for i, directory in enumerate(bkg_list):
   directory.cd()
   hist_list = []
   for key in directory.GetListOfKeys():
       obj = key.ReadObj()
       if isinstance(obj, ROOT.TH1):
           hist_list.append(obj)

   for j, hist_QCD in enumerate(hist_list):
       hist_QCD.SetLineColor(3)
       hist_QCD.SetFillColor(3)
       #hStack.Add(hist_QCD)
       if j == 0 and i == 0:
           hist_background = hist_QCD.Clone("hist_background")
       else:
           hist_background.Add(hist_QCD)

  #Read the signal 1000 hist in direstories
  for i, directory in enumerate(signal_list):
   directory.cd()
   hist_list = []
   for key in directory.GetListOfKeys():
       obj = key.ReadObj()
       if isinstance(obj, ROOT.TH1):
           hist_list.append(obj)

   for j, hist_signal1000 in enumerate(hist_list):
       hist_signal1000.SetLineColor(2)
       hist_signal1000.SetLineWidth(2)
          
  
  #significant = ROOT.TMath.Sqrt(qmu_float(hist_signal1000,hist_background,1.0))

  #print (Cut + Variable + " is ", significant)
  hist_background.Draw("histo")
  hist_background.GetYaxis().SetTitle("Events")
  hist_background.SetMaximum(100000000)
  hist_background.SetMinimum(1)
  hStack.Draw("samehisto")
  #hStack.GetYaxis().SetTitle("Events")
  #hs.SetMaximum(maxy)
  hist_signal1000.Draw("samehisto")
  gPad.RedrawAxis()

  legend = ROOT.TLegend(0.65, 0.75, 0.88, 0.89)
  legend.SetTextSize(0.030)
  legend.SetTextFont(62)
  legend.SetFillColor(0)
  legend.AddEntry(hist_signal1000, "X(1TeV) to Y Y'", "l")
  #legend.AddEntry(hist_signal3000, "X(3TeV) to Y Y'", "l")
  legend.AddEntry(hist_QCD, "QCD", "F")
  legend.SetBorderSize(0)
  legend.Draw()

  canvas.Draw()

  canvas.SaveAs("hstackplots/"+cut+"_"+variable+"_"+category+".pdf")
  print (variable + " done !")


       
def plot_signal_and_bkg(cut,variable,category,signal_list,bkg_list):
  hStack = ROOT.THStack()
  canvas = ROOT.TCanvas()
  canvas.SetLogy()
  for i, directory in enumerate(bkg_list):
   directory.cd()
   hist_list = []
   for key in directory.GetListOfKeys():
       obj = key.ReadObj()
       if isinstance(obj, ROOT.TH1):
           hist_list.append(obj)

   for j, hist_bkg in enumerate(hist_list):
       hist_bkg.SetLineColor(3)
       hist_bkg.SetFillColor(3)
       hStack.Add(hist_bkg)
       if j == 0 and i == 0:
           hist_background = hist_bkg.Clone("hist_background")
       else:
           hist_background.Add(hist_bkg)

  #Read the signal hist in direstories
  for i, directory in enumerate(signal_list):
   directory.cd()
   hist_list = []
   for key in directory.GetListOfKeys():
       obj = key.ReadObj()
       if isinstance(obj, ROOT.TH1):
           hist_list.append(obj)

   for j, hist_signal in enumerate(hist_list):
       hist_signal.SetLineColor(2)
       hist_signal.SetLineWidth(2)
          
  
  max_bkg_bin = hist_background.GetMaximumBin()
  max_bkg_value = hist_background.GetBinContent(max_bkg_bin)

  max_sig_bin = hist_signal.GetMaximumBin()
  max_sig_value = hist_signal.GetBinContent(max_sig_bin)

  Max_value = max_value(max_bkg_value, max_sig_value) 
  
  #significant = ROOT.TMath.Sqrt( 2*( (s+b)*ROOT.TMath.Log(1+(s/b)) - s ) )
  #significant = ROOT.TMath.Sqrt(qmu_float(hist_signal1000,hist_background,1.0))

  #print (Cut + Variable + " is ", significant)
  hist_background.Draw("histo")
  hist_background.GetYaxis().SetTitle("Events")
  hist_background.SetMaximum(Max_value*10)
  hist_background.SetMinimum(1)
  hStack.Draw("samehisto")
  #hStack.GetYaxis().SetTitle("Events")
  #hs.SetMaximum(maxy)
  #hist_signal.Draw("samehisto")
  gPad.RedrawAxis()

  legend = ROOT.TLegend(0.65, 0.75, 0.88, 0.89)
  legend.SetTextSize(0.030)
  legend.SetTextFont(62)
  legend.SetFillColor(0)
  #legend.AddEntry(hist_signal, "X(1TeV) to Y Y'", "l")
  #legend.AddEntry(hist_signal3000, "X(3TeV) to Y Y'", "l")
  legend.AddEntry(hist_bkg, "ttbar", "F")
  legend.SetBorderSize(0)
  legend.Draw()

  canvas.Draw()

  canvas.SaveAs("hstackplots/"+cut+"_"+variable+"_"+category+".pdf")
  print (variable + " done !")
