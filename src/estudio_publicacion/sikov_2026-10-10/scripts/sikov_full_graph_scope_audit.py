"""Truth-free audit of the scope of the full latent graph reconstruction.

This is reporting QA. It never fits or scores evaluation populations and does
not modify the frozen model, protocol, covariates, graphs or selected mixture.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from sikov_model import knn_weights

ROOT=Path(__file__).resolve().parents[1]
ARTICLE=ROOT.parents[1]/'revision_2026-10-05/fuentes/articulo_apuntes_1811.pdf'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    protocol=ROOT/'protocol_freeze.json';frozen=json.loads(protocol.read_text())
    for key in ('statistical_sources_sha256','frozen_inputs_sha256','runtime_binary_sha256'):
        for path,digest in frozen.get(key,{}).items():
            if sha(path)!=digest:raise RuntimeError('Changed frozen file: '+path)
    # Fixed artificial coordinates and responses; these are not experimental
    # populations, do not contain target truth and are never used for selection.
    coords=np.column_stack([np.arange(8,dtype=float),np.zeros(8)])
    sampled=np.array([0,1,3,4,6,7]);unobserved=np.array([2,5])
    WN=knn_weights(coords,k=2);WD=knn_weights(coords[sampled],k=2)
    rho=.4;A=.01;D=np.linspace(.002,.005,len(sampled))
    invN=np.linalg.inv(np.eye(8)-rho*WN);CN=invN@invN.T
    invD=np.linalg.inv(np.eye(6)-rho*WD);CD=invD@invD.T
    X=np.column_stack([np.ones(8),np.arange(8)/7]);y=np.array([.15,.2,.12,.28,.3,.25])
    def prediction(Css,Cross):
        V=A*Css+np.diag(D);Vi=np.linalg.inv(V);Xs=X[sampled]
        beta=np.linalg.solve(Xs.T@Vi@Xs,Xs.T@Vi@y)
        return beta,Xs@beta+A*Cross@Vi@(y-Xs@beta),V
    bn,pn,Vn=prediction(CN[np.ix_(sampled,sampled)],CN[np.ix_(sampled,sampled)])
    bd,pd,Vd=prediction(CD,CD)
    # The directed extension used in the authors' public 2024 preprint has
    # W10=0 and W11=WD. It preserves the sampled-only covariance exactly.
    Wtri=np.zeros((8,8));Wtri[np.ix_(sampled,sampled)]=WD
    for i in unobserved:
        nearest=sampled[np.argsort(abs(coords[sampled,0]-coords[i,0]),kind='stable')[:2]]
        Wtri[i,nearest]=.5
    invT=np.linalg.inv(np.eye(8)-rho*Wtri);CT=invT@invT.T
    checks={
        'full_kNN_has_sampled_to_unobserved_links':bool(np.any(WN[np.ix_(sampled,unobserved)]>0)),
        'full_marginal_differs_from_sampled_kNN_covariance':float(abs(CN[np.ix_(sampled,sampled)]-CD).max())>1e-6,
        'full_marginal_changes_sampling_covariance':float(abs(Vn-Vd).max())>1e-6,
        'fixed_parameter_sampled_predictions_can_differ':float(abs(pn-pd).max())>1e-6,
        'fixed_parameter_GLS_coefficients_can_differ':float(abs(bn-bd).max())>1e-6,
        'directed_W10_zero_preserves_sampled_covariance':float(abs(CT[np.ix_(sampled,sampled)]-CD).max())<1e-12,
        'all_sampled_case_matches_published_covariance':float(abs(CN-invN@invN.T).max())<1e-12,
    }
    if not all(checks.values()):raise RuntimeError(checks)
    result={
        'status':'PASS_SCOPE_LIMITATION_EXPLICIT_FULL_GRAPH_EXTENSION',
        'protocol_sha256':sha(protocol),'article_pdf_sha256':sha(ARTICLE),
        'frozen_sources_modified':False,'evaluation_outcomes_or_truth_read':False,
        'selected_method_or_weights_changed':False,
        'article_evidence':[
            {'printed_page':181,'pdf_page_one_based':11,'section':'2.2.1','finding':'D denotes districts with available survey data; auxiliary X is available in all districts.'},
            {'printed_pages':[185,186],'equations':['2.13','definition of wij'],'finding':'SAR innovations, random effects and W are D-dimensional; W is row standardized over D districts.'},
            {'printed_page':187,'equations':['2.14','2.15','2.16','2.17','2.18'],'finding':'The stated likelihood and SBLUP/SEBLUP formulas use that D-dimensional covariance, without an N-area observation-selection matrix.'},
            {'printed_page':197,'tables':['3.8','3.9','3.10','3.11'],'finding':'Non-sampled district predictions are reported, but no non-sampled predictor or graph augmentation algorithm is specified.'},
        ],
        'authors_2024_source':{
            'publisher':'https://link.springer.com/article/10.1007/s10260-024-00754-0',
            'author_uploaded_preprint':'https://www.researchgate.net/publication/372309772_Prediction_in_non-sampled_areas_under_spatial_small_area_models',
            'preprint_upload_date':'2024-03-05','version_caveat':'Equations below were inspected in the author-uploaded preprint, not inferred from the subscription publisher abstract.',
            'section':2,'equations':[7,8,9,10,11,12],
            'finding':'The preprint extends sampled-area W with W11=W and W10=0. Its non-sampled predictor is X0 beta + (I-rho W00)^(-1) rho W01 uhat_s. This preserves the sampled-area model; the present full kNN marginal model generally permits W10 nonzero and is different.',
        },
        'current_implementation':{
            'latent_graph':'N-area kNN within each poverty stratum',
            'sampled_covariance':'V_s=A*C_N[s,s]+diag(D_s), C_N=(I_N-rho*W_N)^(-1)(I_N-rho*W_N)^(-T)',
            'prediction':'X_all*beta+A*C_N[:,s]*V_s^(-1)*(y_s-X_s*beta)',
            'scope':'Gaussian FH/SAR family and published covariate reconstruction, with an explicit full-graph missing-area extension; not an exact reproduction of the unidentified 2023 numerical implementation.',
            'consequence':'With missing areas, covariance, REML parameter estimates and sampled predictions can differ from a sampled-only SAR graph. This is not merely a rule for extrapolation after fitting.',
        },
        'CRAN_validation_scope':'The numerical sae validation uses all observed areas and validates the common covariance, REML, prediction and bootstrap formulas in that case. It does not validate identity of missing-area graph augmentation with the 2023 authors implementation.',
        'reporting_requirement':'Label the comparator Sikov-family FH/SAR reconstruction; state the full latent graph extension explicitly. Do not call it the exact Sikov 2023 algorithm or claim superiority to his exact published numerical results.',
        'future_comparator_requirement':'A sampled-only SAR with predeclared directed non-sampled prediction rules would be a separate comparator, requiring a new protocol. It must not be added or selected using the present evaluation results.',
        'synthetic_algebra_checks':checks,
        'synthetic_algebra_differences':{
            'max_sampled_covariance':float(abs(CN[np.ix_(sampled,sampled)]-CD).max()),
            'max_fixed_parameter_sampled_prediction':float(abs(pn-pd).max()),
            'max_GLS_beta':float(abs(bn-bd).max()),
            'max_triangular_extension_sampled_covariance':float(abs(CT[np.ix_(sampled,sampled)]-CD).max()),
        },
        'script_sha256':sha(__file__),
    }
    out=ROOT/'validacion/sikov_full_graph_scope_audit.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'checks':checks,'differences':result['synthetic_algebra_differences']},indent=2))

if __name__=='__main__':main()
