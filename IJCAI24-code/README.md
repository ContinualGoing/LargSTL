# The LargeST Benchmark Dataset

Please download and process data from https://github.com/liuxu77/LargeST.


# Run the code

python experiments/LarSTG/main.py --device cuda:2 --dataset CA --years 2019 --model_name LarSTG --seed 2023 --bs 44
python experiments/LarSTG/main.py --device cuda:2 --dataset CA --years 2019 --model_name LarSTG --seed 2024 --bs 44

python experiments/LarSTG/main.py --device cuda:2 --dataset GLA --years 2019 --model_name LarSTG --seed 2023 --bs 64
python experiments/LarSTG/main.py --device cuda:2 --dataset GLA --years 2019 --model_name gwnet --seed 2024 --bs 64

python experiments/LarSTG/main.py --device cuda:2 --dataset GBA --years 2019 --model_name LarSTG --seed 2023 --bs 64
python experiments/LarSTG/main.py --device cuda:2 --dataset GBA --years 2019 --model_name LarSTG --seed 2024 --bs 64



```
