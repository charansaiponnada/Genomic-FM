# Convenience targets (work on the Linux GPU box; on Windows use the commands in README)
.PHONY: install install-gpu test figure longrange ablations pretrain gue report clean

install:      ; pip install -r requirements.txt
install-gpu:  ; pip install -r requirements-gpu.txt
test:         ; python tests/test_core.py
figure:       ; python research_paper/generate_arch_figure.py
longrange:    ; python src/eval_longrange.py --lengths 64 128 256 512 1024 --steps 4000
ablations:    ; python src/ablate.py --seq_len 256 --steps 3000
pretrain:     ; python src/train_mae.py --steps 20000 --window 1024 --d_model 256 --n_layers 6
gue:          ; python src/eval_gue.py --ckpt_dir results/pretrain
report:       ; python src/make_report.py
clean:        ; rm -rf results research_paper/generated_results.tex __pycache__ */__pycache__
