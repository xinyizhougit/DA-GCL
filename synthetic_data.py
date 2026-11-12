from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from config import REGIONS


def create(path, days=8, nodes=4, region_count=4):
    # 合成数据仅验证字段/代码流程，不能用来替代论文结果或原始EXO数据。
    rng = np.random.default_rng(42)
    records = []
    for r, region in enumerate(list(REGIONS)[:region_count]):
        for day in range(days):
            date = pd.Timestamp('2023-09-01')+pd.Timedelta(days=day)
            for slot in range(72):
                departure = date+pd.Timedelta(hours=6, minutes=15*slot)
                for node in range(nodes):
                    stamp = departure+pd.Timedelta(minutes=2*node)
                    records.append(dict(date_operationnelle=str(date.date()), organisation=region,
                        ligne='1', voyage=f'{day}_{slot}', vehicule=f'bus_{r}', sequence_arret=node+1,
                        arret_code=f'{r:02d}{node:04d}', montant=int(rng.poisson(2+4*(7 <= departure.hour < 10))),
                        descendant=int(rng.poisson(2)), date_heure_arrivee_prevue=stamp.isoformat(),
                        date_heure_arrivee_reelle=stamp.isoformat(), date_heure_depart_prevue=stamp.isoformat(),
                        date_heure_depart_reelle=stamp.isoformat(), arret_geom=f'POINT ({-73.6+node*0.002+r*0.01} {45.5+r*0.01})'))
    d = pd.DataFrame(records)
    if len(d) > 50:
        # 保留记录但计数缺失，检查观测掩码是否被正确处理。
        d.loc[25, 'montant'] = np.nan
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(path, index=False)
    return d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', default='synthetic_exo.csv')
    args = parser.parse_args(); create(args.output)
