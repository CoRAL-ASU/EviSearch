# Run from experiment-scripts/ with: bash ../run_baselines.sh
# Or from repo root: cd experiment-scripts && bash ../run_baselines.sh

python experiment-scripts/baseline_file_search.py --pdf "dataset/NCT00104715_Gravis_GETUG_EU'15.pdf" --provider gemini --workers 10
python experiment-scripts/baseline_file_search.py --pdf "dataset/NCT00268476_Attard_STAMPEDE_Lancet'23.pdf" --provider openai --workers 10
python experiment-scripts/baseline_file_search.py --pdf "dataset/NCT00268476_James_STAMPEDE_IJC'22.pdf" --provider openai --workers 10
python experiment-scripts/baseline_file_search.py --pdf "dataset/NCT00309985_Kriayako_CHAARTED_JCO'18.pdf" --provider openai --workers 10
python experiment-scripts/baseline_file_search.py --pdf "dataset/NCT00309985_Sweeney_CHAARTED_NEJM'15.pdf" --provider openai --workers 10
python experiment-scripts/baseline_file_search.py --pdf "dataset/NCT02799602_Hussain_ARASENS_JCO'23.pdf" --provider gemini --model gemini-2.5-flash --workers 10 --reliability-runs 4
