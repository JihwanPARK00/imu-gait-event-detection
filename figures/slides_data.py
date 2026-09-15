"""slides_data.py — 발표 자료용 수치 자동 추출

결과 CSV가 갱신될 때마다 이 스크립트를 다시 돌리면
슬라이드에 들어갈 모든 숫자가 최신으로 갱신된다.

  python slides_data.py            # 화면 출력
  python slides_data.py --md       # results/SLIDES_수치.md 로 저장

읽는 파일
  results/phase1_cells_*.csv       1단계 격자
  results/p2c_*.csv                2단계 교란 (P7 4가지 방식 포함)
  results/dl_p4_*.csv              3단계 딥러닝 교란
  results/trad_ref_raw.csv         전통 기준선 (딥러닝 동일조건)
"""
import sys
import glob
import io
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
OUT = io.StringIO()
MD = '--md' in sys.argv


def p(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    OUT.write(s + '\n')


LV = {'P1': [1, 5, 20, 50, 100], 'P2': [5, 10, 15, 20, 30],
      'P4': [.01, .05, .10, .20, .30], 'P6': [1, 5, 10, 25, 50],
      'P7': [60, 45, 30, 20, 10]}
PAPER = {'P1': 'P1 Noise', 'P2': 'P2 Misalign', 'P4': 'P3 Dropout',
         'P6': 'P4 Bias', 'P7': 'P5 Downsamp'}


def auc(vals):
    return float(np.trapezoid(np.nan_to_num(vals), np.linspace(0, 1, len(vals))))


# ═════════ 슬라이드 3~4: 1단계 격자 ═════════
p('\n## [슬라이드 3-4] 격자 평가 — 어디에, 무엇을, 어떻게')
f1 = glob.glob('results/phase1_cells_*.csv')
if f1:
    d = pd.concat([pd.read_csv(f) for f in f1], ignore_index=True)
    subj = d.groupby(['sensor', 'channel', 'algorithm', 'subject'])[
        ['F1_HS', 'F1_TO']].mean().reset_index()
    cell = subj.groupby(['sensor', 'channel', 'algorithm'])[
        ['F1_HS', 'F1_TO']].agg(['mean', 'std'])
    cell.columns = ['HS_mean', 'HS_sd', 'TO_mean', 'TO_sd']
    cell = cell.reset_index()
    p(f'  격자 규모: {d.trial_id.nunique()} trial x 5 위치 x 9 채널 x 4 알고리즘'
      f' = {d.trial_id.nunique()*180:,} 셀')
    p('\n  Top 5 (4명 평균 HS F1):')
    for _, r in cell.nlargest(5, 'HS_mean').iterrows():
        p(f'    {r.sensor:<12}{r.channel:<12}{r.algorithm:<13}'
          f'{r.HS_mean:.3f} ± {r.HS_sd:.3f}')
    best = cell.nlargest(1, 'HS_mean').iloc[0]
    p(f'\n  → 메시지: 최상위 조합은 {best.sensor} {best.channel} '
      f'{best.algorithm} (F1 {best.HS_mean:.2f})')
    p('  → 상위 5개가 전부 Gyro 채널. Acc/Euler 는 단독으로 부족')

# ═════════ 슬라이드 5: 적응형 임계값 ═════════
    from scipy.stats import wilcoxon
    res = []
    sub = d[d.algorithm.isin(['TB_classic', 'TB_adaptive'])]
    for (s, ch), g in sub.groupby(['sensor', 'channel']):
        pv = g.pivot_table(index='trial_id', columns='algorithm',
                           values=['F1_HS', 'F1_TO'])
        for ev in ['HS', 'TO']:
            a = pv[(f'F1_{ev}', 'TB_adaptive')].values
            c = pv[(f'F1_{ev}', 'TB_classic')].values
            try:
                pval = 1.0 if np.allclose(a, c) else wilcoxon(
                    a, c, alternative='greater').pvalue
            except Exception:
                pval = 1.0
            res.append(dict(sensor=s, channel=ch, event=ev,
                            dF1=a.mean() - c.mean(), p=pval))
    r2 = pd.DataFrame(res)
    p(f'\n## [슬라이드 5] 적응형 임계값 효과')
    p(f'  Wilcoxon 유의 셀: {(r2.p<0.05).sum()} / {len(r2)}')
    p('  최대 개선 5개:')
    for _, r in r2.nlargest(5, 'dF1').iterrows():
        p(f'    {r.sensor:<12}{r.channel:<12}{r.event:<4}'
          f'ΔF1 {r.dF1:+.3f}  p={r.p:.4f}')
    p('  → 메시지: 저진폭 Acc/Euler 채널에서 사용 가능 채널이 확장됨')

# ═════════ 슬라이드 6-8: 2단계 교란 ═════════
f2 = [f for f in glob.glob('results/p2c_*.csv')]
if f2:
    d2 = pd.concat([pd.read_csv(f) for f in f2], ignore_index=True)
    d2['subject'] = d2.trial_id.str.split('_').str[1].str.upper()
    p(f'\n## [슬라이드 6-8] 교란 강건성 ({d2.groupby(["sensor","channel","algorithm"]).ngroups}개 셀)')

    def dr_auc(p7mode):
        df = d2[(d2.p7mode == 'all') | (d2.p7mode == p7mode)].copy()
        base = (df[df.pert_type == 'baseline']
                .groupby(['subject', 'sensor', 'channel', 'algorithm'])
                [['F1_HS', 'F1_TO']].mean()
                .rename(columns={'F1_HS': 'bH', 'F1_TO': 'bT'}).reset_index())
        m = df[df.pert_type != 'baseline'].merge(
            base, on=['subject', 'sensor', 'channel', 'algorithm'])
        m['DR_HS'] = np.where(m.bH > 0.1, np.minimum(m.F1_HS / m.bH, 1.0), np.nan)
        out = {}
        for (a, pt), s in m.groupby(['algorithm', 'pert_type']):
            vals = [s[np.isclose(s.pert_level, l)].DR_HS.mean() for l in LV[pt]]
            out[(a, PAPER[pt])] = auc(vals)
        return out

    fin = dr_auc('native_nyq')
    algos = sorted({k[0] for k in fin})
    p('\n  AUC of DR (HS) — 최종:')
    p('    ' + ' ' * 12 + ''.join(f'{v:>13}' for v in PAPER.values()))
    for a in algos:
        p(f'    {a:<12}' + ''.join(
            f'{fin.get((a,v),float("nan")):>13.2f}' for v in PAPER.values()))
    worst = min(fin.items(), key=lambda x: x[1])
    p(f'\n  → 최약점: {worst[0][0]} / {worst[0][1]} = {worst[1]:.2f}')

    # P7 레벨별
    base = (d2[d2.pert_type == 'baseline']
            .groupby(['subject', 'sensor', 'channel', 'algorithm'])
            .F1_HS.mean().rename('bH').reset_index())
    m7 = d2[d2.pert_type == 'P7'].merge(
        base, on=['subject', 'sensor', 'channel', 'algorithm'])
    m7['DR_HS'] = np.where(m7.bH > 0.1, np.minimum(m7.F1_HS / m7.bH, 1.0), np.nan)
    p('\n  샘플링 레이트별 DR (HS):')
    for mode, label in [('native_nyq', '정확 처리'), ('orig', '참고: 구현오류시')]:
        p(f'    [{label}]')
        for a in algos:
            s = m7[(m7.p7mode == mode) & (m7.algorithm == a)]
            if len(s) == 0:
                continue
            p(f'      {a:<12}' + ''.join(
                f'{s[s.pert_level==l].DR_HS.mean():>8.2f}' for l in LV['P7'])
              + '   (60/45/30/20/10 Hz)')

# ═════════ 슬라이드 9-11: 딥러닝 ═════════
f3 = [f for f in glob.glob('results/dl_p4_*.csv')
      if not f.endswith(('dr.csv', 'curve.csv', 'auc.csv'))]
if f3:
    d3 = pd.concat([pd.read_csv(f) for f in f3], ignore_index=True)
    p('\n## [슬라이드 9-11] 다채널 딥러닝')
    b = (d3[d3.pert_type == 'baseline']
         .groupby(['position', 'model', 'subject']).F1_HS.mean().reset_index())
    bb = b.groupby(['position', 'model']).F1_HS.agg(['mean', 'std'])
    p('\n  baseline F1 (HS, 4명 평균 ± SD):')
    for (pos, mdl), r in bb.iterrows():
        p(f'    {pos:<8}{mdl:<6}{r["mean"]:.3f} ± {r["std"]:.3f}')

    base = (d3[d3.pert_type == 'baseline']
            .groupby(['model', 'position', 'trial_id', 'norm'])
            [['F1_HS', 'F1_TO']].mean()
            .rename(columns={'F1_HS': 'bH', 'F1_TO': 'bT'}).reset_index())
    dd = d3[d3.pert_type != 'baseline'].merge(
        base, on=['model', 'position', 'trial_id', 'norm'])
    dd['DR_HS'] = np.minimum(dd.F1_HS / (dd.bH + 1e-9), 1.0)

    def dl_auc(sub):
        out = {}
        for (pos, mdl, pt), s in sub.groupby(['position', 'model', 'pert_type']):
            vals = [s[np.isclose(s.pert_level, l)].DR_HS.mean() for l in LV[pt]]
            out[(pos, mdl, PAPER[pt])] = auc(vals)
        return out

    for nm in ['adaptive', 'frozen']:
        a9 = dl_auc(dd[(dd.scope == 'all9') & (dd.norm == nm)])
        p(f'\n  AUC of DR (HS) — 정규화 {nm}:')
        p('    ' + ' ' * 16 + ''.join(f'{v:>13}' for v in PAPER.values()))
        for pos in ['shank', 'foot']:
            for mdl in ['CNN', 'LSTM']:
                if (pos, mdl, 'P1 Noise') not in a9:
                    continue
                p(f'    {pos:<8}{mdl:<8}' + ''.join(
                    f'{a9.get((pos,mdl,v),float("nan")):>13.3f}'
                    for v in PAPER.values()))
    p('\n  → 메시지: adaptive 는 거의 전부 1.00 (전처리가 교란을 흡수)')
    p('             frozen 이 실시간 배포에 가까운 값')

    p('\n  교란 채널 수별 AUC (adaptive, HS):')
    for sc, lab in [('ref1', '1채널'), ('gyro3', '3채널'), ('all9', '9채널')]:
        a = dl_auc(dd[(dd.scope == sc) & (dd.norm == 'adaptive')])
        for pos in ['shank']:
            for mdl in ['CNN', 'LSTM']:
                if (pos, mdl, 'P1 Noise') not in a:
                    continue
                p(f'    {lab:<8}{pos:<8}{mdl:<6}' + ''.join(
                    f'{a.get((pos,mdl,v),float("nan")):>10.3f}'
                    for v in PAPER.values()))

if MD:
    with open('results/SLIDES_수치.md', 'w', encoding='utf-8') as fh:
        fh.write('# 발표 자료용 수치 (자동 생성)\n\n```\n')
        fh.write(OUT.getvalue())
        fh.write('\n```\n')
    print('\n[저장] results/SLIDES_수치.md')