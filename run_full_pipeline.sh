# Planning
python experiment-scripts/test_hierarchical_extraction_v2.py \
  --pdf "test_results/new/NCT02799602_Hussain_ARASENS_JCO'23/NCT02799602_Hussain_ARASENS_JCO'23.pdf" \
  --chunks "test_results/new/NCT02799602_Hussain_ARASENS_JCO'23/pdf_chunked.json" \
  --all \
  --workers 10 \
  --name-policy override \
  --output-dir "experiment-scripts/results/NCT02799602_Hussain_ARASENS_JCO'23/extraction_plans"

# Extraction
python experiment-scripts/run_extraction_with_plans_v2.py \
  --pdf "test_results/new/NCT02799602_Hussain_ARASENS_JCO'23/NCT02799602_Hussain_ARASENS_JCO'23.pdf" \
  --chunks "test_results/new/NCT02799602_Hussain_ARASENS_JCO'23/pdf_chunked.json" \
  --provider openai \
  --model gpt-4.1 \
  --workers 10 \
  --name-policy override \
  --output-dir "experiment-scripts/results/NCT02799602_Hussain_ARASENS_JCO'23/extractions"


python experiment-scripts/test_hierarchical_extraction_v2.py \
  --pdf "test_results/new/NCT00268476_Attard_STAMPEDE_Lancet'23/NCT00268476_Attard_STAMPEDE_Lancet'23.pdf" \
  --chunks "test_results/new/NCT00268476_Attard_STAMPEDE_Lancet'23/pdf_chunked.json" \
  --all \
  --workers 10 \
  --name-policy override \
  --output-dir "experiment-scripts/results/NCT00268476_Attard_STAMPEDE_Lancet'23/extraction_plans"

  python experiment-scripts/run_extraction_with_plans_v2.py \
  --pdf "test_results/new/NCT00268476_Attard_STAMPEDE_Lancet'23/NCT00268476_Attard_STAMPEDE_Lancet'23.pdf" \
  --chunks "test_results/new/NCT00268476_Attard_STAMPEDE_Lancet'23/pdf_chunked.json" \
  --provider openai \
  --model gpt-4.1 \
  --workers 10 \
  --name-policy override \
  --output-dir "experiment-scripts/results/NCT00268476_Attard_STAMPEDE_Lancet'23/extractions"
