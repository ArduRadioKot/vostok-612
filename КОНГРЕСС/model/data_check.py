# quick check P_el distribution
els=[float(r['P_el_MW']) for r in rxs if r.get('P_el_MW')]
ths=[float(r['P_th_MW']) for r in rxs if r.get('P_th_MW')]
print('P_el range MW:', min(els), '-', max(els), '| unique:', len(set(els)))
ratios=[t/e for e,t in zip(els,ths) if e and t]
print('P_th/P_el ratio mean:', round(sum(ratios)/len(ratios),2))