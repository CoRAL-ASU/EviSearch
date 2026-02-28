# Run from experiment-scripts/ with: bash ../run_baselines.sh
# Or from repo root: cd experiment-scripts && bash ../run_baselines.sh

# python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT00104715_Gravis_GETUG_EU'15.pdf" --run-eval-only 
# python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT00268476_Attard_STAMPEDE_Lancet'23.pdf" --run-eval-only
# python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT00268476_James_STAMPEDE_IJC'22.pdf" --run-eval-only
# python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT00309985_Kriayako_CHAARTED_JCO'18.pdf" --run-eval-only
# python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT00309985_Sweeney_CHAARTED_NEJM'15.pdf" --run-eval-only
# python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT02799602_Hussain_ARASENS_JCO'23.pdf" --run-eval-only
# python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT01809691_Aggarwal_SWOG1216_JCO'22.pdf" --run-eval-only
# python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT02799602_Smith_ARASENS_NEJM'22.pdf" --run-eval-only
# python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT02446405_Sweeney_ENZAMET_Lancet Onc'23.pdf" --run-eval-only
# python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT01957436_Fizazi_PEACE1_Lancet'22.pdf" --run-eval-only

# # Native Gemini (JSON schema) data extraction – run from repo root with GEMINI_API_KEY set
# python experiment-scripts/baseline_file_search_gemini_native.py --pdf "dataset/NCT00104715_Gravis_GETUG_EU'15.pdf" --model gemini-2.5-flash --workers 10 --run-eval-only
# python experiment-scripts/baseline_file_search_gemini_native.py --pdf "dataset/NCT00268476_James_STAMPEDE_IJC'22.pdf" --model gemini-2.5-flash --workers 10 --run-eval-only
# python experiment-scripts/baseline_file_search_gemini_native.py --pdf "dataset/NCT00309985_Kriayako_CHAARTED_JCO'18.pdf" --model gemini-2.5-flash --workers 10 --run-eval-only
# python experiment-scripts/baseline_file_search_gemini_native.py --pdf "dataset/NCT00309985_Sweeney_CHAARTED_NEJM'15.pdf" --model gemini-2.5-flash --workers 10 --run-eval-only
# python experiment-scripts/baseline_file_search_gemini_native.py --pdf "dataset/NCT01809691_Aggarwal_SWOG1216_JCO'22.pdf" --model gemini-2.5-flash --workers 10 --run-eval-only
# python experiment-scripts/baseline_file_search_gemini_native.py --pdf "dataset/NCT01957436_Fizazi_PEACE1_Lancet'22.pdf" --model gemini-2.5-flash --workers 10 --run-eval-only
# python experiment-scripts/baseline_file_search_gemini_native.py --pdf "dataset/NCT02446405_Sweeney_ENZAMET_Lancet Onc'23.pdf" --model gemini-2.5-flash --workers 10 --run-eval-only
# python experiment-scripts/baseline_file_search_gemini_native.py --pdf "dataset/NCT02799602_Hussain_ARASENS_JCO'23.pdf" --model gemini-2.5-flash --workers 10 --run-eval-only
# python experiment-scripts/baseline_file_search_gemini_native.py --pdf "dataset/NCT02799602_Smith_ARASENS_NEJM'22.pdf" --model gemini-2.5-flash --workers 10 --run-eval-only
# python experiment-scripts/baseline_file_search_gemini_native.py --pdf "dataset/NCT00268476_Attard_STAMPEDE_Lancet'23.pdf" --model gemini-2.5-flash --workers 10 --run-eval-only

# -----------------------------------------------------------------------------
# Baseline: Landing-AI parsed markdown (END-TO-END for all trials)
# Discovers every trial folder that has parsed_markdown.md and runs full pipeline
# (extraction + evaluation) for each trial.
#
# Usage:
#   bash run_baselines.sh                    # Gemini (default)
#   BASELINE=gpt4 bash run_baselines.sh      # GPT-4.1
#
# Optional overrides:
#   MODEL=gemini-2.5-flash WORKERS=10 bash run_baselines.sh
#   BASELINE=gpt4 MODEL=gpt-4.1 bash run_baselines.sh
# -----------------------------------------------------------------------------
BASELINE="${BASELINE:-gemini}"
if [ "$BASELINE" = "gpt4" ]; then
  MODEL="${MODEL:-gpt-4.1}"
  SCRIPT="experiment-scripts/baseline_landing_ai_w_gpt4.py"
  OUT_BASE="experiment-scripts/baseline_landing_ai_w_gpt4/results"
else
  MODEL="${MODEL:-gemini-2.5-flash}"
  SCRIPT="experiment-scripts/baseline_landing_ai_w_gemini.py"
  OUT_BASE="experiment-scripts/baseline_landing_ai_w_gemini/results"
fi
WORKERS="${WORKERS:-10}"
PARSED_ROOT="experiment-scripts/baselines_landing_ai_new_results"

if [ -d "$PARSED_ROOT" ]; then
  echo "Running baseline_landing_ai_w_${BASELINE} end-to-end for all discovered trials"
  echo "Model: $MODEL | Workers: $WORKERS"
  OUT_ROOT="$OUT_BASE/$MODEL"
  for parsed_md in "$PARSED_ROOT"/*/parsed_markdown.md; do
    [ -f "$parsed_md" ] || continue
    trial="$(basename "$(dirname "$parsed_md")")"
    out_dir="$OUT_ROOT/$trial"
    if [ -f "$out_dir/extraction_metadata.json" ]; then
      echo ""
      echo "============================================================"
      echo "Trial: $trial — SKIP (already processed)"
      echo "============================================================"
      continue
    fi
    echo ""
    echo "============================================================"
    echo "Trial: $trial"
    echo "============================================================"
    python "$SCRIPT" \
      --trial "$trial" \
      --model "$MODEL" \
      --workers "$WORKERS" || echo "FAILED trial: $trial (continuing)"
    echo "Sleeping 60s to avoid rate limits..."
    sleep 60
  done
fi
