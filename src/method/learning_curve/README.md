# Learning-curve experiment

Esperimento per misurare la performance del modello al variare del numero di
sequenze di training. Test e validation rimangono fissi (`test.txt`, `val.txt`).

## Struttura

```
learning_curve/
├── make_subsets.py        # genera i sottoinsiemi nested di train.txt
├── run_one.py             # un singolo run (N, seed): train + predict + evaluate
├── aggregate.py           # raccoglie tutti i run → CSV + plot
├── subsets/               # sottoinsiemi generati di train.txt
└── runs/N{N}_seed{S}/     # output per singolo run
    ├── checkpoints/
    ├── outputs/           # predizioni per-sequenza
    ├── evaluation/        # summary_avg.{txt,csv}, fp50_x_wp_avg.png, ...
    └── wp_metric.json     # |wp_pred − wp_GT| calcolato qui
```

I path canonici `src/method/{checkpoints,outputs,evaluation}/` del run full
**non vengono toccati** — vengono monkey-patchati su `method.config` solo per
la durata del run.

## Design

- **Griglia N**: `{4, 8, 12, 16, 24, 32, 40, 46}` (8 punti).
- **Schema seed K(N)**: `10/8/8/6/4/3/3/1` (totale **43 run**). Più
  ripetizioni a N piccoli, dove la varianza tra sottoinsiemi è massima e il
  training è più rumoroso (early-stopping su val piccola). A N=46 basta un
  seed: tutti i sottoinsiemi coincidono con `train.txt`.
- **Iperparametri**: `--batch-size 64 --lr 4e-4` (linear scaling rule rispetto
  al baseline di config.py: bs=16, lr=1e-4 → bs=64, lr=4e-4). Passati via CLI
  nei launcher per non modificare `config.py`.
- **Nested**: per ogni seed la lista di training viene permutata UNA volta;
  il subset di taglia N è il prefisso di taglia N di quella permutazione.
  Quindi `subset(N=4, seed=0) ⊂ subset(N=8, seed=0) ⊂ ... ⊂ subset(N=46, seed=0)`.
- **Metrica primaria**: `|avg_pred_wp − avg_gt_wp|`, calcolata come la media
  per-sequenza del wp al frame `fp50` (frame in cui la cumulativa supera il
  50% dell'area finale), come fa `evaluate._save_avg_fp50_x_wp_plot`.
- **Metriche secondarie**: IoU/Dice/F1 sui frame "evento" e fp50_pred/fp50_gt.

## Come si lancia

### 1) Genera i sottoinsiemi (una sola volta)

```bash
python src/method/learning_curve/make_subsets.py
```

Produce 8 × 3 = 24 file in `subsets/`.

### 2) Lancia i run

Tre modi, dal più automatico al più manuale.

**(a) Tutto in sequenza, set-and-forget** (~48h, ~2 giorni):
```bash
bash src/method/learning_curve/launchers/run_all.sh
```
Lancia N=4 → N=8 → ... → N=46 nell'ordine. Se interrompi a metà, al rilancio
salta i run già finiti grazie alla resumability.

**(b) Un blocco N alla volta** (utile per fermarsi tra un N e l'altro):
```bash
bash src/method/learning_curve/launchers/run_N4.sh    # ~3h  (5 seed)
bash src/method/learning_curve/launchers/run_N8.sh    # ~4h  (4 seed)
bash src/method/learning_curve/launchers/run_N12.sh   # ~4.5h (3 seed)
# ... eccetera fino a run_N46.sh
```
Ogni launcher salva un log per-run in `logs/N{N}_seed{S}.log`.

**(c) Singolo run manuale** (debug o re-run mirato):
```bash
python src/method/learning_curve/run_one.py --n 8 --seed 1
```

**Resumability**: se `runs/N{N}_seed{S}/evaluation/summary_avg.csv` esiste già,
il run viene saltato. Usa `--force` su `run_one.py` per riallenare comunque.

**Argomenti extra a train.py** (epochs, lr, batch-size, ...) — passabili in
tutti i modi:
```bash
python src/method/learning_curve/run_one.py --n 8 --seed 0 --epochs 100
bash src/method/learning_curve/launchers/run_N8.sh -- --epochs 100
bash src/method/learning_curve/launchers/run_all.sh -- --batch-size 8
```

### 3) Aggrega i risultati

A run terminati (anche parzialmente — aggrega quello che trova):

```bash
python src/method/learning_curve/aggregate.py
```

Produce:
- `learning_curve.csv` — una riga per `(N, seed)` con tutte le metriche.
- `learning_curve.png` — due pannelli: errore wp e IoU, con error bar (std sui seed).

## Tempo totale stimato

43 run con BS=64 (~3× più veloce di BS=16 a parità di epoche). Tempo di
training approssimativamente lineare in N.

| N  | seed | tempo/run | totale blocco |
|----|------|-----------|---------------|
| 4  | 10   | ~15 min   | ~2.5h |
| 8  | 8    | ~20 min   | ~3h   |
| 12 | 8    | ~30 min   | ~4h   |
| 16 | 6    | ~40 min   | ~4h   |
| 24 | 4    | ~1h       | ~4h   |
| 32 | 3    | ~1.5h     | ~4h   |
| 40 | 3    | ~1.5h     | ~5h   |
| 46 | 1    | ~2h       | ~2h   |
|    |      | **totale** | **~28h** |

Le stime sono indicative. Il tempo dipende anche da quando early stopping
scatta (più presto a N piccoli).

## Note

- L'`fp50` torna `-1` se la sequenza non raggiunge mai il 50% di embolia
  predetta/GT. In quel caso non contribuisce alla media wp (filtrato silenziosamente).
- Il calcolo della wp richiede `data/SenecioIVERdroughtVCs.xlsx` (già usato
  da `evaluate.py`). Se manca, `wp_metric.json` conterrà solo `null`.
- `train.py` usa `torch.manual_seed(config.SEED)` con `SEED=42` fissato.
  Per avere variabilità tra i seed dell'esperimento (0/1/2) la differenza
  arriva solo dal *sottoinsieme di training*, non dall'init del modello.
  Va bene per misurare la varianza dovuta alla scelta dei dati — che è la
  fonte di rumore dominante a N piccoli.
