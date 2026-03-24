from functools import partial
import numpy as np
import awkward as ak
import logging
import vector
from copy import copy

vector.register_awkward()

import pepper
from pepper.processor_basic import VariationArg

logger = logging.getLogger(__name__)

class ProcessorTTXS(pepper.ProcessorTTbarLL):

    def _check_config_integrity(self, config):
        """Check integrity of configuration file."""

        super()._check_config_integrity(config)

    def process_selection(self, selector, dsname, is_mc, filler):
        era = self.get_era(selector.data, is_mc)

        if not is_mc:
            selector.add_cut("Era", partial(self.has_era, era), no_callback=True)

        #if dsname.startswith("TTTo"):
        #    selector.set_column("gent_lc", self.gentop, lazy=True)
        #    if "top_pt_reweighting" in self.config:
        #        selector.add_cut(
        #            "Top pt reweighting", self.do_top_pt_reweighting,
        #            no_callback=False)
       
        if self.config["compute_systematics"] \
            and self.config["do_generator_uncertainties"] and is_mc:
            self.add_generator_uncertainies(dsname, selector)
        if is_mc:
            selector.add_cut(
                "Cross section", partial(self.crosssection_scale, dsname))

        if "blinding_denom" in self.config:
            selector.add_cut("Blinding", partial(self.blinding, is_mc))
        selector.add_cut("Lumi", partial(self.good_lumimask, is_mc, dsname))

        pos_triggers, neg_triggers = pepper.misc.get_trigger_paths_for(
            dsname, is_mc, self.config["dataset_trigger_map"],
            self.config["dataset_trigger_order"], era=era)
        selector.add_cut("Trigger", partial(
            self.passing_trigger, pos_triggers, neg_triggers))
        
        #selector.add_cut("MET filters", partial(self.met_filters, is_mc))
     
        selector.add_cut("No add leps",
                         partial(self.no_additional_leptons, is_mc))
        selector.set_column("Electron", self.pick_electrons)
        selector.set_column("Muon", self.pick_muons)
        selector.set_column("Lepton", partial(
            self.build_lepton_column, is_mc, selector.rng))
        '''
        # Wait with hists filling after channel masks are available
        selector.add_cut("At least 2 leps", partial(self.lepton_pair, is_mc),
                         no_callback=True)
        #selector.add_column("")
        selector.set_cat("channel", {"is_ee", "is_em", "is_mm"})
        selector.set_multiple_columns(self.channel_masks)

        selector.add_cut("Opposite sign", self.opposite_sign_lepton_pair)
        selector.add_cut("Chn trig match",
                         partial(self.channel_trigger_matching, era))
        if "trigger_sfs_tnp" in self.config and is_mc:
            selector.add_cut("Trigger SFs", self.apply_trigger_sfs_tnp_sl)
        selector.add_cut("Req lep pT", self.lep_pt_requirement)
        '''
        if (is_mc and self.config["compute_systematics"]
                and self.config["do_jet_variations"]
                and dsname not in self.config["dataset_for_systematics"]):
            if hasattr(filler, "sys_overwrite"):
                assert filler.sys_overwrite is None
            for variarg in self.get_jetmet_variation_args():
                selector_copy = copy(selector)
                filler.sys_overwrite = variarg.name
                self.process_selection_jet_part(selector_copy, is_mc,
                                                variarg, dsname, filler, era)
                if self.eventdir is not None:
                    logger.debug(f"Saving per event info for variation"
                                 f" {variarg.name}")
                    self.save_per_event_info(
                        dsname + "_" + variarg.name, selector_copy, False)
            filler.sys_overwrite = None

        # Do normal, no-variation run
        self.process_selection_jet_part(selector, is_mc,
                                        self.get_jetmet_nominal_arg(),
                                        dsname, filler, era)
        logger.debug("Selection done")


    def process_selection_jet_part(self, selector, is_mc, variation, dsname,
                                   filler, era):
        logger.debug(f"Running jet_part with variation {variation.name}")

        reapply_jec = ("reapply_jec" in self.config
                       and self.config["reapply_jec"])
        selector.set_multiple_columns(partial(
            self.compute_jet_factors, is_mc, era, reapply_jec, variation.junc, variation.jer, selector.rng))
        selector.set_column("OrigJet", selector.data["Jet"])
        selector.set_column("Jet", partial(self.build_jet_column, is_mc))
        selector.set_column("FatJet", partial(self.build_fatjet_column, is_mc))
        selector.set_column("HT", self.sum_ak4jet_pt)
        if is_mc:
          selector.set_column("genWeight", self.Get_event_weight)
        #selector.add_cut("Has_at_least_two_fatjets", self.At_least_two_fatjets)
        #selector.add_cut("Has_at_least_one_genfatjet", self.At_least_one_genfatjet)
        #selector.add_cut("Has_at_least_two_genjets", self.At_least_two_genjets)
        selector.add_cut("Has_at_least_one_fatjet", self.At_least_one_fatjet)
        selector.set_multiple_columns(partial(self.Jet_categories, selector))
        selector.add_cut("Has_at_least_two_jets", self.At_least_two_jets)
        #selector.add_cut("Jet_and_FatJet_DeltaR_cut", self.DeltaR_cut)
        #selector.set_column("two_fatjets_mass", self.mass_two_fatjets)
        if is_mc:
          selector.set_column("GenFatJet", self.build_gen_fatjet_column)
        #selector.set_column("GenFatJet_Y_mass", self.mass_leading_genjets)
        #selector.set_column("GenJet_Y_mass", self.mass_leading_genfatjets)
        #selector.set_column("GenJet_Y_mass", self.Gen_Y_mass)
        #selector.set_column("GenJet_Yprime_mass", self.Gen_Yprime_mass)
        #selector.set_column("GenJet_X_mass", self.Gen_X_mass)
        #selector.set_column("Reco_Y_mass", self.Reco_Y_mass)
        selector.set_column("Reco_Yprime_mass", self.Reco_Yprime_mass)
        selector.set_column("Reco_X_mass", partial(self.Reco_X_mass,dsname) )
        selector.add_cut("HT_cut", self.HT_cut)
        #selector.set_column("Reco_Y_pt", self.Reco_Y_pt)
        #selector.set_column("Reco_Yprime_pt", self.Reco_Yprime_pt)
        #selector.set_column("Reco_X_pt", self.Reco_X_pt)
        #selector.set_column("multisample_select_X_mass", self.multisample_select_X_mass)
        selector.set_column("Particle_Net_Xqq_vs_QCD_score", self.Particle_Net_Xqq_score)
        selector.set_column("Reco_DeltaR_jet1_fatjet1", self.DeltaR_jet1_fatjet1)
        selector.set_column("Reco_DeltaR_jet2_fatjet1", self.DeltaR_jet2_fatjet1)
        #selector.set_column("Reco_DeltaR_jet3_fatjet1", self.DeltaR_jet3_fatjet1)
        #selector.set_column("Gen_DeltaR_jet1_fatjet1", self.DeltaR_jet1_fatjet1)
        #selector.set_column("Gen_DeltaR_jet2_fatjet1", self.DeltaR_jet2_fatjet1)
        #selector.set_column("GenP_DeltaPhi_YYprime", self.GenP_DeltaPhi_YYprime)
        selector.add_cut("Xqq_vs_QCD_VLoose_WP", self.Xqq_vs_QCD_VLoose)
        selector.add_cut("Xqq_vs_QCD_Loose_WP", self.Xqq_vs_QCD_Loose)
        selector.add_cut("Xqq_vs_QCD_Medium_WP", self.Xqq_vs_QCD_Medium)
        selector.add_cut("Xqq_vs_QCD_Tight_WP", self.Xqq_vs_QCD_Tight)
        #selector.set_column("Delta_phi_YYprime", self.Delta_phi_YYprime)
        #selector.set_column("Reco_Y_pt", self.Reco_Y_pt)
        #selector.set_column("Reco_Y", self.Reco_Y)
        #selector.set_column("GenFatJet_match_Y", self.Gen_match_Y_fatjet)
        #selector.set_column("GenFatJet_match_Yprime", self.Gen_match_Yprime_fatjet)
        #selector.set_column("GenYprimeMass_decay_top", self.Gen_Yprime_decay_top)
        #selector.set_column("GenYprimeMass_decay_otherquark", self.Gen_Yprime_decay_otherquark)
        #selector.set_column("Y mass", self.construct_Y_mass)
        #selector.set_column("Yprime mass", self.construct_Yprime_mass)
        #selector.set_column("Gen Y mass", self.Gen_Y_mass)
        #selector.set_column("Gen Yprime mass", self.Gen_Yprime_mass)
        #selector.set_column("nbtag", self.num_btags)

        #selector.set_multiple_columns(partial(self.btag_categories, selector))
        '''
        selector.set_column("MET", partial( 
            self.build_met_column, is_mc, variation.junc, variation.jer, selector.rng, era, variation="central"))
        '''
        #selector.add_cut("Has jet(s)", self.has_jets)
        
        #selector.add_cut("Has btag(s)", partial(self.btag_cut, is_mc))
        

    def has_era(self, era, data):
        if era == "no_era":
            return np.full(len(data), False)
        else:
            return np.full(len(data), True)

    def num_btags(self, data):
        jets = data["Jet"]
        nbtags = ak.sum(jets["btagged"], axis=1)
        return ak.where(ak.num(jets) > 0, nbtags, 0)
    '''
    def num_jets(self, data):
        jets = data["Jet"]
        njets = ak.num(jets)
        return njets
    '''
    def Jet_categories(self, selector, data):
        #HP is PN larger than tight WP, LP is PN between tight and medium WP
        cats = {} 
        Xqq_vs_QCD = data["FatJet"][:,0].particleNetMD_Xqq/(data["FatJet"][:,0].particleNetMD_Xqq + data["FatJet"][:,0].particleNetMD_QCD)        
        cats["HP"] = (Xqq_vs_QCD>=0.891)
        cats["LP"] = ((Xqq_vs_QCD<0.891) & (Xqq_vs_QCD>0.810))
        cats["rest"] = (Xqq_vs_QCD<0.810)

        selector.set_cat("Categories", {"HP", "LP", "rest"})
        return cats       

    def btag_categories(self, selector, data):
        cats = {}
        num_btagged = data["nbtag"]

        cats["b1"] = (num_btagged == 1)
        cats["b2+"] = (num_btagged >= 2)

        selector.set_cat("btags", {"b1", "b2+"})

        return cats

    def Get_event_weight(self,data):
        weight = data["genWeight"]
        return weight

    def apply_trigger_sfs_tnp_sl(self, data):
        leps = data["Lepton"]
        is_dl = ak.num(leps) > 1

        trigger_sfs = self.config["trigger_sfs_tnp"]

        abseta = abs(leps.eta)
        pt = leps.pt

        sources = ["EleMC","EleData","MuMC","MuData"]
        vars = [("nominal", None)]
        sfs_vars = {}
        if self.config["compute_systematics"]:
            vars.extend([(var,d) for var in sources for d in ["up", "down"]])

        for var, d in vars:

            effs_e_mc = trigger_sfs[("e", "eff_mc")](abseta=abseta, pt=pt, variation=(d if var == "EleMC" else "central"))
            effs_e_data = trigger_sfs[("e", "eff_data")](abseta=abseta, pt=pt, variation=(d if var == "EleData" else "central"))
            effs_m_mc = trigger_sfs[("mu", "eff_mc")](abseta=abseta, pt=pt, variation=(d if var == "MuMC" else "central"))
            effs_m_data = trigger_sfs[("mu", "eff_data")](abseta=abseta, pt=pt, variation=(d if var == "MuData" else "central"))

            effs_mc = ak.where(abs(leps.pdgId) == 11, effs_e_mc, effs_m_mc)
            eff_event_mc_dl = ak.sum(effs_mc, axis=1) - ak.prod(effs_mc, axis=1)
            eff_event_mc_sl = effs_mc[:,0]
            eff_event_mc = ak.where(is_dl, eff_event_mc_dl, eff_event_mc_sl)
                
            effs_data = ak.where(abs(leps.pdgId) == 11, effs_e_data, effs_m_data)
            
            eff_event_data_dl = ak.sum(effs_data, axis=1) - ak.prod(effs_data, axis=1)
            eff_event_data_sl = effs_data[:,0]
            eff_event_data = ak.where(is_dl, eff_event_data_dl, eff_event_data_sl)
        
            sfs_event = eff_event_data / eff_event_mc
            sfs_vars[(var, d)] = sfs_event

        sfs_nom = sfs_vars[("nominal", None)]
        if self.config["compute_systematics"]:
            systs = {f"triggersf{var}": (sfs_vars[(var, "up")] / sfs_nom, sfs_vars[(var, "down")] / sfs_nom) for var in sources}
            return sfs_nom, systs
        else:
            return sfs_nom

    def At_least_one_genfatjet(self, data):
        """select events that contain at least 1 gen level fatjets"""
        accept = np.asarray(ak.num(data["GenJetAK8"]) >= 1) 
        return accept

    def At_least_two_genjets(self, data):
        """select events that contain at least 2 gen level jets"""
        accept = np.asarray(ak.num(data["GenJet"]) >= 2) 
        return accept

    def At_least_one_fatjet(self, data):
        """select events that contain at least 1 reco level fatjets"""
        #breakpoint()
        accept = np.asarray(ak.num(data["FatJet"]) >= 1) 
        return accept

    def At_least_two_jets(self, data):
        """select events that contain at least 2 reco level jets"""
        accept = np.asarray(ak.num(data["Jet"]) >= 2) 
        return accept

    def DeltaR_cut(self, data):
        deltaR_1 = np.sqrt( (data["Jet"][:,0].phi-data["FatJet"][:,0].phi)**2 + (data["Jet"][:,0].eta-data["FatJet"][:,0].eta)**2 ) 
        deltaR_2 = np.sqrt( (data["Jet"][:,1].phi-data["FatJet"][:,0].phi)**2 + (data["Jet"][:,1].eta-data["FatJet"][:,0].eta)**2 )
        accept = np.asarray(deltaR_2 > 0.8)
        return accept

    def Gen_Y_mass(self, data):
        return (data["GenJetAK8"][:, 0]).mass 

    def Gen_Yprime_mass(self, data):
        jet1_eta, jet1_phi = data["GenJet"][:,0].eta, data["GenJet"][:,0].phi
        jet2_eta, jet2_phi = data["GenJet"][:,1].eta, data["GenJet"][:,1].phi
        
        fatjet1_eta, fatjet1_phi = data["GenJetAK8"][:,0].eta, data["GenJetAK8"][:,0].phi

        deltaR_jet1_fatjet1 = self.delta_r(jet1_eta, jet1_phi, fatjet1_eta, fatjet1_phi)  
        deltaR_jet2_fatjet1 = self.delta_r(jet2_eta, jet2_phi, fatjet1_eta, fatjet1_phi)

       
        num_ak4 = ak.num(data["GenJet"]) 
        padded_events = ak.pad_none(data["GenJet"], 5, axis=1)


        #Gen_Yprime_mass = np.where(((deltaR_jet1_fatjet1>=0.8) & (deltaR_jet2_fatjet1>=0.8) & (ak.num(data["GenJet"]) >= 2)), (padded_events[:, 0] + padded_events[:, 1]).mass, -99)

        #Gen_Yprime_mass = np.where(((deltaR_jet1_fatjet1<=0.8) & (deltaR_jet2_fatjet1>=0.8) & (ak.num(data["GenJet"]) >= 3)), (padded_events[:, 1] + padded_events[:, 2]).mass, -99)

        #Gen_Yprime_mass = np.where(((deltaR_jet1_fatjet1>=0.8) & (deltaR_jet2_fatjet1<=0.8) & (ak.num(data["GenJet"]) >= 3)), (padded_events[:, 0] + padded_events[:, 2]).mass, -99)

        Gen_Yprime_mass = np.where(((deltaR_jet1_fatjet1<=0.8) & (deltaR_jet2_fatjet1>=0.8) & (ak.num(data["GenJet"]) >= 3)), (padded_events[:, 1] + padded_events[:, 2]).mass, (padded_events[:, 0] + padded_events[:, 2]).mass)

        #Gen_Yprime_mass = np.where(
        #    ((deltaR_jet1_fatjet1>=0.8) & (deltaR_jet2_fatjet1>=0.8) & (ak.num(data["Jet"]) >= 2)),
        #    (padded_events[:, 0] + padded_events[:, 1]).mass,
        #    np.where(
        #        ((deltaR_jet1_fatjet1<=0.8) & (deltaR_jet2_fatjet1>=0.8) & (ak.num(data["Jet"]) >= 3)),
        #        (padded_events[:, 1] + padded_events[:, 2]).mass,
        #        -99, 
        #    )
        #)

        return Gen_Yprime_mass 

    def Gen_X_mass(self, data):
        jet1_eta, jet1_phi = data["GenJet"][:,0].eta, data["GenJet"][:,0].phi
        jet2_eta, jet2_phi = data["GenJet"][:,1].eta, data["GenJet"][:,1].phi
        
        fatjet1_eta, fatjet1_phi = data["GenJetAK8"][:,0].eta, data["GenJetAK8"][:,0].phi

        deltaR_jet1_fatjet1 = self.delta_r(jet1_eta, jet1_phi, fatjet1_eta, fatjet1_phi)  
        deltaR_jet2_fatjet1 = self.delta_r(jet2_eta, jet2_phi, fatjet1_eta, fatjet1_phi)

       
        num_ak4 = ak.num(data["GenJet"]) 
        padded_events = ak.pad_none(data["GenJet"], 5, axis=1)

        Gen_X_mass = np.where(((deltaR_jet1_fatjet1<=0.8) & (deltaR_jet2_fatjet1>=0.8) & (ak.num(data["GenJet"]) >= 3)), (padded_events[:, 1] + padded_events[:, 2] + data["GenJetAK8"][:, 0]).mass, (padded_events[:, 0] + padded_events[:, 2] + data["GenJetAK8"][:, 0]).mass)

        return Gen_X_mass 
    
    def GenP_DeltaPhi_YYprime(self, data):
        # pick Y and Y' in gen particle
        part = data["GenPart"]
        Gen_Y_particle = part[ (part.pdgId == 23) ]
        Gen_Yprime_particle = part[ (part.pdgId == 25) ]
        return self.delta_phi(Gen_Y_particle.phi, Gen_Yprime_particle.phi)


    def DeltaR_jet1_fatjet1(self, data):
        jet1_eta, jet1_phi = data["Jet"][:,0].eta, data["Jet"][:,0].phi
        fatjet1_eta, fatjet1_phi = data["FatJet"][:,0].eta, data["FatJet"][:,0].phi

        #jet1_eta, jet1_phi = data["GenJet"][:,0].eta, data["GenJet"][:,0].phi
        #fatjet1_eta, fatjet1_phi = data["GenJetAK8"][:,0].eta, data["GenJetAK8"][:,0].phi

        deltaR_jet1_fatjet1 = self.delta_r(jet1_eta, jet1_phi, fatjet1_eta, fatjet1_phi)  
        return deltaR_jet1_fatjet1

    def DeltaR_jet2_fatjet1(self, data):
        jet2_eta, jet2_phi = data["Jet"][:,1].eta, data["Jet"][:,1].phi
        fatjet1_eta, fatjet1_phi = data["FatJet"][:,0].eta, data["FatJet"][:,0].phi

        #jet2_eta, jet2_phi = data["GenJet"][:,1].eta, data["GenJet"][:,1].phi
        #fatjet1_eta, fatjet1_phi = data["GenJetAK8"][:,0].eta, data["GenJetAK8"][:,0].phi

        deltaR_jet2_fatjet1 = self.delta_r(jet2_eta, jet2_phi, fatjet1_eta, fatjet1_phi)
        return deltaR_jet2_fatjet1

    def DeltaR_jet3_fatjet1(self, data):
        padded_events = ak.pad_none(data["Jet"], 5, axis=1)
        jet3_eta, jet3_phi = padded_events[:,2].eta, padded_events[:,2].phi

        #jet2_eta, jet2_phi = data["Jet"][:,1].eta, data["Jet"][:,1].phi
        fatjet1_eta, fatjet1_phi = data["FatJet"][:,0].eta, data["FatJet"][:,0].phi

        #jet2_eta, jet2_phi = data["GenJet"][:,1].eta, data["GenJet"][:,1].phi
        #fatjet1_eta, fatjet1_phi = data["GenJetAK8"][:,0].eta, data["GenJetAK8"][:,0].phi

        deltaR_jet3_fatjet1 = self.delta_r(jet3_eta, jet3_phi, fatjet1_eta, fatjet1_phi)
        return deltaR_jet3_fatjet1

    def Delta_phi_YYprime(self, data):
        jet1_eta, jet1_phi = data["Jet"][:,0].eta, data["Jet"][:,0].phi
        jet2_eta, jet2_phi = data["Jet"][:,1].eta, data["Jet"][:,1].phi
        
        fatjet1_eta, fatjet1_phi = data["FatJet"][:,0].eta, data["FatJet"][:,0].phi

        deltaR_jet1_fatjet1 = self.delta_r(jet1_eta, jet1_phi, fatjet1_eta, fatjet1_phi)  
        deltaR_jet2_fatjet1 = self.delta_r(jet2_eta, jet2_phi, fatjet1_eta, fatjet1_phi)

        padded_events = ak.pad_none(data["Jet"], 5, axis=1)
        Yprime_phi = np.where(((deltaR_jet1_fatjet1<=0.8) & (deltaR_jet2_fatjet1>=0.8) & (ak.num(data["Jet"]) >= 3)), (padded_events[:, 1] + padded_events[:, 2]).phi, -99)
        
        return self.delta_phi(Yprime_phi, fatjet1_phi)

    def Reco_Y_mass(self, data):
        #reco_fatjet = ak.pad_none(data["FatJet"], 2, axis=1)
        #abs_diff_from_Y_col1 = np.abs(reco_fatjet[:, 0].mass - 60)
        #abs_diff_from_Y_col2 = np.abs(reco_fatjet[:, 1].mass - 60)

        #Y_mass = np.where(abs_diff_from_Y_col1 < abs_diff_from_Y_col2, reco_fatjet[:, 0].mass, reco_fatjet[:, 1].mass)
        
        Y_mass = (data["FatJet"][:, 0]).mass
        return Y_mass     
    '''
    def Reco_Yprime_mass(self, data):
        padded_events = ak.pad_none(data["Jet"], 5, axis=1)
        #jet1_eta, jet1_phi = data["Jet"][:,0].eta, data["Jet"][:,0].phi
        #jet2_eta, jet2_phi = data["Jet"][:,1].eta, data["Jet"][:,1].phi
        #jet3_eta, jet3_phi = data["Jet"][:,2].eta, data["Jet"][:,2].phi
        jet1_eta, jet1_phi = padded_events[:,0].eta, padded_events[:,0].phi
        jet2_eta, jet2_phi = padded_events[:,1].eta, padded_events[:,1].phi
        jet3_eta, jet3_phi = padded_events[:,2].eta, padded_events[:,2].phi
        
        fatjet1_eta, fatjet1_phi = data["FatJet"][:,0].eta, data["FatJet"][:,0].phi

        deltaR_jet1_fatjet1 = self.delta_r(jet1_eta, jet1_phi, fatjet1_eta, fatjet1_phi)  
        deltaR_jet2_fatjet1 = self.delta_r(jet2_eta, jet2_phi, fatjet1_eta, fatjet1_phi)
        deltaR_jet3_fatjet1 = self.delta_r(jet3_eta, jet3_phi, fatjet1_eta, fatjet1_phi)

       
        #num_ak4 = ak.num(data["Jet"]) 
        

        #Yprime_mass = np.where(((deltaR_jet1_fatjet1>=0.8) & (deltaR_jet2_fatjet1>=0.8) & (ak.num(data["Jet"]) >= 2)), (padded_events[:, 0] + padded_events[:, 1]).mass, -99)

        Yprime_mass = np.where(((deltaR_jet1_fatjet1<=0.8) & (deltaR_jet2_fatjet1>=0.8) & (ak.num(data["Jet"]) >= 3)), (padded_events[:, 1] + padded_events[:, 2]).mass, -99)

        #Yprime_mass = np.where(((deltaR_jet1_fatjet1>=0.8) & (deltaR_jet2_fatjet1<=0.8) & (deltaR_jet3_fatjet1>=0.8) & (ak.num(data["Jet"]) >= 3)), (padded_events[:, 0] + padded_events[:, 2]).mass, -99)

        #Yprime_mass = np.where(((deltaR_jet1_fatjet1<=0.8) & (deltaR_jet2_fatjet1>=0.8) & (ak.num(data["Jet"]) >= 3)), (padded_events[:, 1] + padded_events[:, 2]).mass, (padded_events[:, 0] + padded_events[:, 2]).mass)

        #Yprime_mass = np.where(
        #    ((deltaR_jet1_fatjet1>=0.8) & (deltaR_jet2_fatjet1>=0.8) & (ak.num(data["Jet"]) >= 2)),
        #    (padded_events[:, 0] + padded_events[:, 1]).mass,
        #    np.where(
        #        ((deltaR_jet1_fatjet1<=0.8) & (deltaR_jet2_fatjet1>=0.8) & (ak.num(data["Jet"]) >= 3)),
        #        (padded_events[:, 1] + padded_events[:, 2]).mass,
        #        -99, 
        #    )
        #)

        #Yprime_mass = np.where(
        #    ((deltaR_jet1_fatjet1>=0.8) & (deltaR_jet2_fatjet1>=0.8) & (ak.num(data["Jet"]) >= 2)),
        #    (padded_events[:, 0] + padded_events[:, 1]).mass,
        #    np.where(
        #        ((deltaR_jet1_fatjet1<=0.8) & (deltaR_jet2_fatjet1>=0.8) & (ak.num(data["Jet"]) >= 3)),
        #        (padded_events[:, 1] + padded_events[:, 2]).mass,
        #        np.where(
        #            ((deltaR_jet1_fatjet1<=0.8) & (deltaR_jet2_fatjet1<=0.8) & (ak.num(data["Jet"]) >= 4)),
        #            (padded_events[:, 2] + padded_events[:, 3]).mass,
        #            -99,
        #        )
        #    )
        #)

        return Yprime_mass
    '''
    def Reco_Yprime_mass(self, data):
        n_events = len(data["FatJet"])

        # 初始化输出数组（全部填充 NaN）
        invariant_mass = np.full(n_events, np.nan)  # 默认值为 NaN

        fatjet = data["FatJet"][:, 0]
        fatjet_p4 = vector.awk(
          ak.zip({
          "pt": fatjet.pt,
          "eta": fatjet.eta,
          "phi": fatjet.phi,
          "mass": fatjet.mass
          }, with_name="Momentum4D")
        )
        padded_events = ak.pad_none(data["Jet"], 4, axis=1)
        jets = padded_events[:, :4]
        jets_p4 = vector.awk(
          ak.zip({
          "pt": jets.pt,
          "eta": jets.eta,
          "phi": jets.phi,
          "mass": jets.mass
          }, with_name="Momentum4D")
        )

        # 计算每个jet与leading fatjet的deltaR
        delta_eta = jets_p4.eta - ak.broadcast_arrays(fatjet_p4.eta, jets_p4.eta)[0]
        delta_phi = (jets_p4.phi - ak.broadcast_arrays(fatjet_p4.phi, jets_p4.phi)[0] + np.pi) % (2 * np.pi) - np.pi
        delta_R = np.sqrt(delta_eta**2 + delta_phi**2)

        # 选择deltaR > 0.8的jets
        selected_jets = jets_p4[delta_R > 0.8]

        # 确保每个事件中至少有两个满足条件的jets
        event_mask = ak.num(selected_jets) >= 2
        valid_events = ak.where(event_mask)[0]  # 获取有效事件的索引

        if len(valid_events) > 0:
        # 取前两个满足条件的jets
          jet1 = selected_jets[event_mask][:, 0]
          jet2 = selected_jets[event_mask][:, 1]
          valid_mass = (jet1 + jet2).mass

          # 将有效值填充到输出数组中
          invariant_mass[valid_events] = ak.to_numpy(valid_mass)  # 转换为 NumPy 并填充

        # 计算这两个jets的不变质量
        #combined_p4 = jet1 + jet2
        #print("this step is ok 6")
        #Yprime_mass = combined_p4.mass
        #Yprime_mass = np.where(mask, combined_p4.mass , -99)
        
        return ak.Array(invariant_mass)  # 返回与输入事件数对齐的数组
        #return Yprime_mass
    '''
    def Reco_X_mass(self, dsname, data):
        jet1_eta, jet1_phi = data["Jet"][:,0].eta, data["Jet"][:,0].phi
        jet2_eta, jet2_phi = data["Jet"][:,1].eta, data["Jet"][:,1].phi
        
        fatjet1_eta, fatjet1_phi = data["FatJet"][:,0].eta, data["FatJet"][:,0].phi

        deltaR_jet1_fatjet1 = self.delta_r(jet1_eta, jet1_phi, fatjet1_eta, fatjet1_phi)  
        deltaR_jet2_fatjet1 = self.delta_r(jet2_eta, jet2_phi, fatjet1_eta, fatjet1_phi)

       
        num_ak4 = ak.num(data["Jet"]) 
        padded_events = ak.pad_none(data["Jet"], 4, axis=1)

        X_mass = np.where(((deltaR_jet1_fatjet1<=0.8) & (deltaR_jet2_fatjet1>=0.8) & (ak.num(data["Jet"]) >= 3)), (padded_events[:, 1] + padded_events[:, 2] + data["FatJet"][:, 0]).mass, (padded_events[:, 0] + padded_events[:, 2] + data["FatJet"][:, 0]).mass)
        return X_mass
    '''
    def Reco_X_mass(self, dsname, data):
        n_events = len(data["FatJet"])

        # 初始化输出数组（全部填充 NaN）
        invariant_mass = np.full(n_events, np.nan)  # 默认值为 NaN

        fatjet = data["FatJet"][:, 0]
        fatjet_p4 = vector.awk( 
          ak.zip({
          "pt": fatjet.pt,
          "eta": fatjet.eta,
          "phi": fatjet.phi,
          "mass": fatjet.mass
          }, with_name="Momentum4D")
        )
        padded_events = ak.pad_none(data["Jet"], 4, axis=1)
        jets = padded_events[:, :4]
        jets_p4 = vector.awk(
          ak.zip({
          "pt": jets.pt,
          "eta": jets.eta,
          "phi": jets.phi,
          "mass": jets.mass
          }, with_name="Momentum4D")
        )
        # 计算每个jet与leading fatjet的deltaR
        delta_eta = jets_p4.eta - ak.broadcast_arrays(fatjet_p4.eta, jets_p4.eta)[0]
        delta_phi = (jets_p4.phi - ak.broadcast_arrays(fatjet_p4.phi, jets_p4.phi)[0] + np.pi) % (2 * np.pi) - np.pi
        delta_R = np.sqrt(delta_eta**2 + delta_phi**2)

        # 选择deltaR > 0.8的jets
        selected_jets = jets_p4[delta_R > 0.8]
        #selected_fatjet = fatjet_p4[delta_R > 0.8]

        # 确保每个事件中至少有两个满足条件的jets
        event_mask = ak.num(selected_jets) >= 2
        valid_events = ak.where(event_mask)[0]  # 获取有效事件的索引

        if len(valid_events) > 0:
        # 取前两个满足条件的jets
          jet1 = selected_jets[event_mask][:, 0]
          jet2 = selected_jets[event_mask][:, 1]
          fatjet1 = fatjet_p4[event_mask]
          valid_mass = (jet1 + jet2 + fatjet1).mass

          # 将有效值填充到输出数组中
          invariant_mass[valid_events] = ak.to_numpy(valid_mass)  # 转换为 NumPy 并填充

        # 计算这两个jets的不变质量
        #combined_p4 = jet1 + jet2
        #print("this step is ok 6")
        #Yprime_mass = combined_p4.mass
        #Yprime_mass = np.where(mask, combined_p4.mass , -99)
        
        return ak.Array(invariant_mass)  # 返回与输入事件数对齐的数组

    def Reco_Yprime_pt(self, data):
        jet1_eta, jet1_phi = data["Jet"][:,0].eta, data["Jet"][:,0].phi
        jet2_eta, jet2_phi = data["Jet"][:,1].eta, data["Jet"][:,1].phi
        
        fatjet1_eta, fatjet1_phi = data["FatJet"][:,0].eta, data["FatJet"][:,0].phi

        deltaR_jet1_fatjet1 = self.delta_r(jet1_eta, jet1_phi, fatjet1_eta, fatjet1_phi)  
        deltaR_jet2_fatjet1 = self.delta_r(jet2_eta, jet2_phi, fatjet1_eta, fatjet1_phi)

       
        num_ak4 = ak.num(data["Jet"]) 
        padded_events = ak.pad_none(data["Jet"], 4, axis=1)

        Yprime_pt = np.where(((deltaR_jet1_fatjet1<=0.8) & (deltaR_jet2_fatjet1>=0.8) & (ak.num(data["Jet"]) >= 3)), (padded_events[:, 1] + padded_events[:, 2]).pt, (padded_events[:, 0] + padded_events[:, 2]).pt)

        return Yprime_pt

    def Reco_X_pt(self, data):
        jet1_eta, jet1_phi = data["Jet"][:,0].eta, data["Jet"][:,0].phi
        jet2_eta, jet2_phi = data["Jet"][:,1].eta, data["Jet"][:,1].phi
        
        fatjet1_eta, fatjet1_phi = data["FatJet"][:,0].eta, data["FatJet"][:,0].phi

        deltaR_jet1_fatjet1 = self.delta_r(jet1_eta, jet1_phi, fatjet1_eta, fatjet1_phi)  
        deltaR_jet2_fatjet1 = self.delta_r(jet2_eta, jet2_phi, fatjet1_eta, fatjet1_phi)

       
        num_ak4 = ak.num(data["Jet"]) 
        padded_events = ak.pad_none(data["Jet"], 4, axis=1)

        X_pt = np.where(((deltaR_jet1_fatjet1<=0.8) & (deltaR_jet2_fatjet1>=0.8) & (ak.num(data["Jet"]) >= 3)), (padded_events[:, 1] + padded_events[:, 2] + data["FatJet"][:, 0]).pt, (padded_events[:, 0] + padded_events[:, 2] + data["FatJet"][:, 0]).pt)

        return X_pt

    def Reco_Y_pt(self, data):

        return (data["FatJet"][:, 0]).pt

    def Reco_Y(self, data):
        jet1_eta, jet1_phi = data["Jet"][:,0].eta, data["Jet"][:,0].phi
        jet2_eta, jet2_phi = data["Jet"][:,1].eta, data["Jet"][:,1].phi
        
        fatjet1_eta, fatjet1_phi = data["FatJet"][:,0].eta, data["FatJet"][:,0].phi

        deltaR_jet1_fatjet1 = self.delta_r(jet1_eta, jet1_phi, fatjet1_eta, fatjet1_phi)  
        deltaR_jet2_fatjet1 = self.delta_r(jet2_eta, jet2_phi, fatjet1_eta, fatjet1_phi)

       
        num_ak4 = ak.num(data["Jet"]) 
        padded_events = ak.pad_none(data["Jet"], 4, axis=1)

        Y = data["FatJet"]
        #Y = np.where(
        #    (deltaR_jet1_fatjet1>=0.8) & (deltaR_jet2_fatjet1>=0.8) & (ak.num(data["Jet"]) >= 2),
        #    data["FatJet"],
        #    np.where(
        #        (deltaR_jet1_fatjet1<=0.8) & (deltaR_jet2_fatjet1>=0.8) & (ak.num(data["Jet"]) >= 3),
        #        data["FatJet"],
        #        np.where(
        #            (deltaR_jet1_fatjet1<=0.8) & (deltaR_jet2_fatjet1<=0.8) & (ak.num(data["Jet"]) >= 4),
        #            data["FatJet"],
        #            None,
        #        )
        #    )
        #)

        return Y


    def construct_Y_mass(self, data):
        #calculate the mass with 90GV
        #abs_diff_from_Y_col1 = np.abs(data["FatJet"][:, 0].mass - 90)
        #abs_diff_from_Y_col2 = np.abs(data["FatJet"][:, 1].mass - 90)

        abs_diff_from_Y_col1 = np.abs(data["FatJet"][:, 0].mass - 90)
        abs_diff_from_Y_col2 = np.abs(data["FatJet"][:, 1].mass - 90)

        Y_mass = np.where(abs_diff_from_Y_col1 < abs_diff_from_Y_col2, data["FatJet"][:, 0].mass, data["FatJet"][:, 1].mass)
        
        return Y_mass     

    def construct_Yprime_mass(self, data):
        #calculate the mass with 500GV
        abs_diff_from_Y_col1 = np.abs(data["FatJet"][:, 0].mass - 500)
        abs_diff_from_Y_col2 = np.abs(data["FatJet"][:, 1].mass - 500)

        Yprime_mass = np.where(abs_diff_from_Y_col1 < abs_diff_from_Y_col2, data["FatJet"][:, 0].mass, data["FatJet"][:, 1].mass)
        return Yprime_mass     

    def sum_ak4jet_pt(self,data):
        jets_pt = data["Jet"].pt 
        HT = ak.sum(jets_pt,axis=1)
        return HT

    def HT_cut(self,data):
        jets_pt = data["Jet"].pt 
        HT = ak.sum(jets_pt,axis=1)
        accept = HT>1200 
        return accept
         
    def build_gen_fatjet_column(self, data):
        gen_fatjet = data["GenJetAK8"]
        return gen_fatjet
        
    def Gen_match_Y_fatjet(self, data):
        #pick Gen Y particle
        part = data["GenPart"]
        Gen_Y_particle = part[ (part.pdgId == 23) ]
        abs_diff_from_Y_col2 = np.abs(data["FatJet"][:, 1].mass - 500)

        Yprime_mass = np.where(abs_diff_from_Y_col1 < abs_diff_from_Y_col2, data["FatJet"][:, 0].mass, data["FatJet"][:, 1].mass)
        return Yprime_mass     

    
    def build_gen_fatjet_column(self, data):
        gen_fatjet = data["GenJetAK8"]
        return gen_fatjet
        
    def Gen_match_Y_fatjet(self, data):
        #pick Gen Y particle
        part = data["GenPart"]
        Gen_Y_particle = part[ (part.pdgId == 23) ]
        #Do the march
        gen_fatjet = data["GenJetAK8"]
        deltaR = np.sqrt( (Gen_Y_particle[:,0].phi-gen_fatjet[:,0].phi)**2 + (Gen_Y_particle[:,0].eta-gen_fatjet[:,0].eta)**2 )
        Gen_Match_Y_fatjet = np.where(deltaR<0.4, gen_fatjet[:,0].mass, -99)
        return Gen_Match_Y_fatjet       

    def Gen_match_Yprime_fatjet(self, data):
        #pick Gen Yprime particle
        part = data["GenPart"]
        Gen_Yprime_particle = part[ (part.pdgId == 25) ]
        #Do the march
        gen_fatjet = data["GenJetAK8"]
        deltaR = np.sqrt( (Gen_Yprime_particle[:,0].phi-gen_fatjet[:,1].phi)**2 + (Gen_Yprime_particle[:,0].eta-gen_fatjet[:,1].eta)**2 )
        Gen_Match_Yprime_fatjet = np.where(deltaR<0.4, gen_fatjet[:,1].mass, -99)
        return Gen_Match_Yprime_fatjet       

    def Gen_Yprime_decay_top(self, data):
        gen_fatjet = data["GenJetAK8"]
        part = data["GenPart"]
        part = part[ part.pdgId == 25 ]
        daughter = part.children
        Yprime_decay_top_mass = np.where( daughter.pdgId == 6, gen_fatjet[:,1].mass, -99)
        return Yprime_decay_top_mass[:,0]

    def Gen_Yprime_decay_otherquark(self, data):
        gen_fatjet = data["GenJetAK8"]
        part = data["GenPart"]
        part = part[ part.pdgId == 25 ]
        daughter = part.children
        Yprime_decay_otherquark_mass = np.where( ( daughter.pdgId==5 ) | ( daughter.pdgId==4 ) | ( daughter.pdgId==3 ) | ( daughter.pdgId==2 ) | ( daughter.pdgId==1 ) , gen_fatjet[:,1].mass, -99)
        return Yprime_decay_otherquark_mass[:,0]

    def delta_r(self, eta1, phi1, eta2, phi2):
      delta_eta = eta1 - eta2
      delta_phi = np.abs(phi1 - phi2)
      delta_phi = ak.where(delta_phi > np.pi, 2*np.pi - delta_phi, delta_phi)
      return np.sqrt(delta_eta**2 + delta_phi**2)

    def delta_phi(self, phi1, phi2):
      delta_phi = np.abs(phi1 - phi2)
      delta_phi = ak.where(delta_phi > np.pi, 2*np.pi - delta_phi, delta_phi)
      return delta_phi

        #part = part[ abs(part.pdgId) == 6 ]
        #mother = part.parent
        #if np.all( abs(mother.pdgId) != 25 ):
        #   return False
        #else :
        #   return True

    
    def At_least_four_jets(self, data):
        """select events that contain at least four jets"""
        accept = np.asarray(ak.num(data["Jet"]) >= 4) 
        return accept

    def Particle_Net_Xqq_score(self, data):

        return (data["FatJet"][:,0].particleNetMD_Xqq/(data["FatJet"][:,0].particleNetMD_Xqq + data["FatJet"][:,0].particleNetMD_QCD)) 
    

    def Xqq_vs_QCD_VLoose(self, data):
        Xqq_vs_QCD = data["FatJet"][:,0].particleNetMD_Xqq/(data["FatJet"][:,0].particleNetMD_Xqq + data["FatJet"][:,0].particleNetMD_QCD)
        VLoose = np.asarray( Xqq_vs_QCD >= 0.4 )
        return VLoose

    def Xqq_vs_QCD_Loose(self, data):
        Xqq_vs_QCD = data["FatJet"][:,0].particleNetMD_Xqq/(data["FatJet"][:,0].particleNetMD_Xqq + data["FatJet"][:,0].particleNetMD_QCD)
        Loose = np.asarray( Xqq_vs_QCD >= 0.579 )
        return Loose

    def Xqq_vs_QCD_Medium(self, data):
        Xqq_vs_QCD = data["FatJet"][:,0].particleNetMD_Xqq/(data["FatJet"][:,0].particleNetMD_Xqq + data["FatJet"][:,0].particleNetMD_QCD)
        Medium = np.asarray( Xqq_vs_QCD >= 0.810 )
        return Medium

    def Xqq_vs_QCD_Tight(self, data):
        Xqq_vs_QCD = data["FatJet"][:,0].particleNetMD_Xqq/(data["FatJet"][:,0].particleNetMD_Xqq + data["FatJet"][:,0].particleNetMD_QCD)
        Tight = np.asarray( Xqq_vs_QCD >= 0.891 )
        return Tight
    
    def multisample_select_X_mass(self, data):
        reco_fatjet = ak.pad_none(data["FatJet"], 2, axis=1)
        #reco_fatjet = data["FatJet"]
        #part = data["GenPart"]
        #part_X = part[ part.pdgId == 9000001 ]
        #part_Y = part[ part.pdgId == 23 ]
        #part_Yprime = part[ part.pdgId == 25 ]
        X_mass = (reco_fatjet[:,0] + reco_fatjet[:,1]).mass

        #X_mass = np.where( ((part_X.mass==3000) & (part_Y.mass==80) & (part_Yprime.mass==80) & (ak.num(data["FatJet"]) >= 2)) , (reco_fatjet[:,0]).mass, -99)
        #return part_X.mass
        return X_mass

    #def construct_Y_mass(self, data):
    #    chi_2_Y_mass_ini= 999999
    #    Y_mass = -99
    #    for i in range(4):
    #      for j in range(i,4):
    #        chi_2_Y_mass = ((data["Jet"][:, i] + data["Jet"][:, j]).mass - 90) ** 2

if __name__ == "__main__":
    from pepper import runproc
    runproc.run_processor(
        ProcessorTTXS,
        "Select events from nanoAODs using the TTXS processor."
        "This will save cutflows, histograms and, if wished, per-event data. "
        "Histograms are saved in a Coffea format and are ready to be plotted "
        "by plot_control.py")
