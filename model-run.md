uv run python -m fault_localization.experiments.cli train --config config/gru.yaml --seed 42

uv run python -m fault_localization.experiments.cli train --config config/rgcn.yaml --seed 42

uv run python -m fault_localization.experiments.cli train --config config/rgsage_mean.yaml --seed 42

uv run python -m fault_localization.experiments.cli train --config config/rgsage_max.yaml --seed 42

uv run python -m fault_localization.experiments.cli train --config config/rgatv2.yaml --seed 42

for model in gru rgcn rgsage_mean rgsage_max rgatv2; do
  for seed in 42 43 44 45 46; do
    uv run python -m fault_localization.experiments.cli train \
      --config "config/${model}.yaml" \
      --seed "$seed"
  done
done


uv run python -m fault_localization.experiments.cli report \
  --benchmark benchmark_v1.0