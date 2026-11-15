"""Reproducible command line. No web, no absolute local paths, no silent defaults."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .data import read_sources, diagnose,load_scenario
from .ranking import build_ranking
from .expert import load_independent_labels, loo_evaluate, train_and_export, load_trained_model, predict_model
from .labeling import blank_template, aggregate_ratings
from .provenance import cost_profile, load_workbook_case, source_consistency
from .modern import export_comparison

ROOT=Path(__file__).resolve().parents[1]


def main(argv=None):
    parser=argparse.ArgumentParser(description='Exploratory hybrid SMR/site decision-support pipeline')
    sub=parser.add_subparsers(dest='cmd',required=True)
    v=sub.add_parser('validate',help='Inspect input data quality')
    v.add_argument('--root',type=Path,default=ROOT)
    c=sub.add_parser('source-audit',help='Audit workbook totals and cross-document source inconsistencies')
    c.add_argument('--root',type=Path,default=ROOT)
    t=sub.add_parser('evaluate-expert',help='LOO-validation of GP on independently collected labels')
    t.add_argument('--root',type=Path,default=ROOT)
    t.add_argument('--labels',type=Path,required=True)
    cm=sub.add_parser('compare-models',help='Compare GPR, BayesianRidge and SVR on independent site labels; TabPFN-3.5 opt-in')
    cm.add_argument('--root',type=Path,default=ROOT)
    cm.add_argument('--labels',type=Path,required=True)
    cm.add_argument('--tabpfn',action='store_true',help='Opt in to TabPFN-3.5; requires separate installation, checkpoint and license')
    cm.add_argument('--out',type=Path,default=ROOT/'modern_model_benchmark')
    lt=sub.add_parser('labels-template', help='Generate empty independent expert rating form')
    lt.add_argument('--root',type=Path,default=ROOT)
    lt.add_argument('--experts',type=int,default=3)
    lt.add_argument('--out',type=Path,default=ROOT/'expert_ratings_blank.csv')
    al=sub.add_parser('aggregate-labels', help='Aggregate documented, independent expert ratings')
    al.add_argument('--root',type=Path,default=ROOT)
    al.add_argument('--ratings',type=Path,required=True)
    al.add_argument('--min-experts',type=int,default=2)
    al.add_argument('--out',type=Path,default=ROOT/'independent_labels.csv')
    tr=sub.add_parser('train-expert',help='Fit GPR on real independent labels and export validated artifact')
    tr.add_argument('--root',type=Path,default=ROOT)
    tr.add_argument('--labels',type=Path,required=True)
    tr.add_argument('--out',type=Path,default=ROOT/'trained_expert')
    pr=sub.add_parser('predict-expert',help='Inference using a trusted model.joblib artifact')
    pr.add_argument('--root',type=Path,default=ROOT)
    pr.add_argument('--model',type=Path,required=True)
    pr.add_argument('--out',type=Path,default=ROOT/'expert_predictions.csv')
    b=sub.add_parser('run',help='Scenario ranking + uncertainty stress test')
    b.add_argument('--root',type=Path,default=ROOT)
    b.add_argument('--scenario',type=Path,default=ROOT/'data'/'scenario_demo.csv')
    b.add_argument('--labels',type=Path,help='Optional independently collected expert labels')
    b.add_argument('--model',type=Path,help='Trusted model created by train-expert (alternative to --labels)')
    b.add_argument('--rank-stability-samples',type=int,default=100,help='Dirichlet weight sensitivity samples')
    b.add_argument('--reactor',type=str,default='СВБР-100',help='Substring match, or ALL for catalogue')
    b.add_argument('--max-modules',type=int,default=3)
    b.add_argument('--max-reactor-mw',type=float,default=350.)
    b.add_argument('--samples',type=int,default=250)
    b.add_argument('--seed',type=int,default=42)
    b.add_argument('--cost-profile',choices=['bigstone_2026','legacy_2026'],default='bigstone_2026',
                   help='Big Stone project-estimate anchor (default), or historical illustrative anchors')
    b.add_argument('--operation-life-override',type=int,help='Optional explicitly assumed operation life in years (e.g. thesis 60 vs catalog 50)')
    b.add_argument('--out',type=Path,default=ROOT/'results')
    args=parser.parse_args(argv)
    sites,reactors=read_sources(args.root)
    if args.cmd=='source-audit':
        _,profile=cost_profile('bigstone_2026')
        print(json.dumps({'source_case':load_workbook_case(),'profile':profile,
                          'inconsistencies':source_consistency(sites,reactors)},ensure_ascii=False,indent=2)); return
    if args.cmd=='validate':
        report={'n_sites':len(sites),'n_reactors':len(reactors),'diagnostics':diagnose(sites,reactors)}
        print(json.dumps(report,ensure_ascii=False,indent=2));return
    if args.cmd=='evaluate-expert':
        labeled=load_independent_labels(args.labels,sites)
        print(json.dumps(loo_evaluate(labeled),ensure_ascii=False,indent=2));return
    if args.cmd=='compare-models':
        labeled=load_independent_labels(args.labels,sites)
        result=export_comparison(labeled,args.out,include_tabpfn=args.tabpfn)
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    if args.cmd=='labels-template':
        args.out.parent.mkdir(parents=True,exist_ok=True)
        blank_template(sites,args.experts).to_csv(args.out,index=False,encoding='utf-8-sig')
        print('EMPTY expert rating template:',args.out);return
    if args.cmd=='aggregate-labels':
        labels,audit=aggregate_ratings(args.ratings,sites,args.min_experts)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        labels.to_csv(args.out,index=False,encoding='utf-8-sig')
        audit.to_csv(args.out.with_name(args.out.stem+'_audit.csv'),index=False,encoding='utf-8-sig')
        print('Real expert labels:',args.out,'sites:',len(labels));return
    if args.cmd=='train-expert':
        labeled=load_independent_labels(args.labels,sites)
        meta=train_and_export(sites,labeled,args.out,args.labels)
        print(json.dumps({'output_dir':str(args.out),'n_train_sites':meta['n_train_sites'],
                          'validation':meta['validation']},ensure_ascii=False,indent=2));return
    if args.cmd=='predict-expert':
        model,meta=load_trained_model(args.model)
        mu,std=predict_model(sites,model)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame({'site':sites.site,'expert_prediction':mu,
                      'expert_std_uncalibrated':std}).to_csv(args.out,index=False,encoding='utf-8-sig')
        print('Predictions:',args.out,'trained sites:',meta['n_train_sites']);return
    scenario=load_scenario(args.scenario,sites)
    selected=None if args.reactor.lower()=='all' else args.reactor
    assumptions, profile_details=cost_profile(args.cost_profile)
    rows,excluded,stats=build_ranking(sites,reactors,scenario,reactor_filter=selected,
          max_modules=args.max_modules,max_reactor_mw=args.max_reactor_mw,
          samples=args.samples,seed=args.seed,independent_labels_path=args.labels,
          independent_model_path=args.model,rank_stability_samples=args.rank_stability_samples,
          p=assumptions,cost_profile_name=args.cost_profile,
          operation_life_override=args.operation_life_override)
    stats['cost_profile_source_details']=profile_details
    stats['source_diagnostics']=diagnose(sites,reactors)
    stats['scenario_file']=str(args.scenario)
    stats['scenario_data_statuses']=scenario['data_status'].value_counts().to_dict()
    args.out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out/'ranking.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(excluded,columns=['site','reactor','modules','reason']).to_csv(args.out/'excluded.csv',index=False,encoding='utf-8-sig')
    (args.out/'report.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"Screened: {stats['n_candidates']}, excluded: {stats['n_excluded']}, Pareto: {stats.get('n_on_pareto_front',0)}")
    print('Files:',args.out/'ranking.csv',args.out/'excluded.csv',args.out/'report.json')
    if rows:
        for x in rows[:5]:
            print(f"  {x['rank']:2}. {x['site'][:29]:29} {x['reactor'][:15]:15} n={x['modules']}  "
                  f"NPV={x['npv_bln']:+.1f} bln  Score={x['composite_0_100']:.1f} [UNVERIFIED]")

if __name__=='__main__':main()
