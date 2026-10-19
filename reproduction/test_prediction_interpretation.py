"""Synthetic mechanics only; no research data, truth, fits or historical MCSE."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import prediction_interpretation as helper
from article_printed_cells import mismatches

class DiagnosticMechanics(unittest.TestCase):
    def test_constant_chains_are_undefined_and_between_chain_failure_visible(self):
        x=np.zeros((4000,2));x[:,1]=np.repeat(np.arange(4),1000)
        result=helper.batch_mcse(x)
        self.assertTrue(np.isnan(result['mcse']).all())
        self.assertTrue(np.isnan(result['classic_split_Rhat'][0]))
        self.assertTrue(np.isinf(result['classic_split_Rhat'][1]))

    def test_equal_chain_weighting_has_analytic_batch_variance(self):
        x=np.tile(np.repeat(np.arange(20,dtype=float),50),4)[:,None]
        result=helper.batch_mcse(x)
        expected=np.sqrt(np.var(np.arange(20),ddof=1)/(4*20))
        self.assertAlmostEqual(result['mcse'][0],expected,14)
        self.assertEqual(result['draws_used'],4000)
        self.assertEqual(result['batch_length'],50)

    def test_different_support_and_nonfinite_mismatches_survive(self):
        with self.assertRaises(ValueError):helper.compare_array(np.zeros(2),np.zeros(3),1e-10)
        row=helper.compare_array(np.array([0.,np.nan,np.inf]),np.array([.01,np.nan,-np.inf]),1e-10)
        self.assertEqual(row['exact_cells'],1);self.assertEqual(row['tolerance_cells'],1)
        self.assertEqual(row['nonfinite_mismatches'],1)

    def test_complete_discrepancy_payload_is_not_truncated(self):
        fresh=[[str(i)]for i in range(40)];article=[['different']for _ in range(39)]
        rows=mismatches(fresh,article)
        self.assertEqual(len(rows),40);self.assertEqual(rows[-1]['row'],40)
        self.assertIsNone(rows[-1]['article'])

    def test_full_cohort_gate_retains_invalid_truth_role_and_missing_world(self):
        with tempfile.TemporaryDirectory(prefix='iaos_metadata_fixture_')as directory:
            root=Path(directory);worlds=list(helper.paths(root/'new',root/'reference'))
            self.assertEqual(len(worlds),322)
            gate=helper.cohort_gate(worlds,'protocol');self.assertFalse(gate['complete'])
            for world in worlds:
                for role,methods in (('original',helper.worker.METHODS[:9]),('published',helper.worker.METHODS[9:])):
                    path=world[role]/'completion.json';path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_text(json.dumps(dict(protocol_sha256='protocol',truth_read_by_fits=False,convergence=dict.fromkeys(methods,True))))
                if 'aggregate'in world:
                    (world['aggregate']/'completion.json').write_text(json.dumps(dict(protocol_sha256='protocol',truth_read_by_fits=False,convergence=dict.fromkeys(helper.worker.METHODS,True))))
            self.assertTrue(helper.cohort_gate(worlds,'protocol')['complete'])
            path=worlds[0]['original']/'completion.json';record=json.loads(path.read_text());record['truth_read_by_fits']=True;path.write_text(json.dumps(record))
            self.assertFalse(helper.cohort_gate(worlds,'protocol')['complete'])
            path.unlink();self.assertTrue(helper.cohort_gate(worlds,'protocol')['pending'])

    def test_conditional_mean_variance_and_fixed_mixture_mcse_reconstruct(self):
        source=Path(__file__).resolve().parents[1]/'src/estudio_publicacion'
        for folder in (source/'scripts/competitors',source/'sikov_2026-10-10/scripts'):
            sys.path.insert(0,str(folder))
        import beta_power
        from improve_candidates import mixture_prediction
        rng=np.random.default_rng(8675309) # Synthetic diagnostic fixture, not a production seed.
        beta=rng.normal(0,.2,(4000,1));gamma=np.zeros((4000,1));logk=np.full(4000,np.log(40.))
        p=helper.expit(beta[:,0]);a=np.repeat((40*p)[:,None],3,axis=1);b=40-a
        conditional=a/(a+b);pdraw=rng.beta(a,b)
        mean=conditional.mean(axis=0);variance=(a*b/((a+b)**2*(a+b+1))).mean(axis=0)+conditional.var(axis=0)
        icc=dict(mean=np.column_stack((mean,mean)),lower=np.column_stack((np.quantile(pdraw,.025,axis=0),)*2),
            upper=np.column_stack((np.quantile(pdraw,.975,axis=0),)*2),mse=np.column_stack((variance,variance)))
        sar=dict(mean=np.full((3,2),.45),mse=np.full((3,2),.005))
        selected=mixture_prediction(dict(p_mean=icc['mean'],p_variance=icc['mse'],marginal_draws=[pdraw,pdraw]),sar,.75)
        trial=dict(attempt=1,max_Rhat=1.001,min_bulk_ESS=1000.,min_tail_ESS=1000.,E_BFMI=[.8]*4,divergences=0,max_treedepth_hits=0)
        d=dict(X=np.ones((3,1)),y=np.empty((0,2)),n_eff=np.empty((0,2)),sample_idx=np.array([],dtype=int),basis=np.zeros((3,1)))
        with tempfile.TemporaryDirectory(prefix='iaos_mcse_fixture_')as directory:
            folder=Path(directory);np.savez_compressed(folder/'ICC_conditional_draws.npz',anemia_marginal=pdraw,stunting_marginal=pdraw)
            world=dict(original=folder,real=False);predictions=dict(ICC=icc,Sikov_SAR=sar,Selected=selected)
            status=dict(success=True,sampler_trials=[trial])
            with patch.object(helper,'retained',return_value=(beta,gamma,logk,trial)),\
                    patch.object(beta_power,'working_icc',return_value=(0.,{})),\
                    patch.object(beta_power,'cluster_effective_n',return_value=(np.array([]),{})):
                arrays,receipt=helper.mcse_endpoint(world,0,d,None,status,predictions,predictions,{},20,2)
            self.assertTrue(np.isfinite(arrays[('ICC','mean')]).all())
            np.testing.assert_allclose(arrays[('Selected','mean')],.75*arrays[('ICC','mean')],atol=0,rtol=0)
            self.assertTrue(np.isfinite(arrays[('ICC','lower')]).all())
            self.assertTrue(np.isfinite(arrays[('Selected','upper')]).all())
            self.assertLess(max(receipt['prediction_reconstruction_max_errors'].values()),1e-12)

if __name__=='__main__':unittest.main()
