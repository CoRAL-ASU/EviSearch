# Run from experiment-scripts/ with: bash ../run_baselines.sh
# Or from repo root: cd experiment-scripts && bash ../run_baselines.sh

python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT00104715_Gravis_GETUG_EU'15.pdf" --run-eval-only 
python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT00268476_Attard_STAMPEDE_Lancet'23.pdf" 
python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT00268476_James_STAMPEDE_IJC'22.pdf" 
python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT00309985_Kriayako_CHAARTED_JCO'18.pdf" 
python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT00309985_Sweeney_CHAARTED_NEJM'15.pdf" 
python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT02799602_Hussain_ARASENS_JCO'23.pdf"