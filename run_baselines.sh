# Run from experiment-scripts/ with: bash ../run_baselines.sh
# Or from repo root: cd experiment-scripts && bash ../run_baselines.sh

# python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT00104715_Gravis_GETUG_EU'15.pdf" --run-eval-only 
# python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT00268476_Attard_STAMPEDE_Lancet'23.pdf" 
# python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT00268476_James_STAMPEDE_IJC'22.pdf" 
# python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT00309985_Kriayako_CHAARTED_JCO'18.pdf" 
# python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT00309985_Sweeney_CHAARTED_NEJM'15.pdf" 
# python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT02799602_Hussain_ARASENS_JCO'23.pdf"



# python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT01809691_Aggarwal_SWOG1216_JCO'22.pdf"
# python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT02799602_Smith_ARASENS_NEJM'22.pdf"
# python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT02446405_Sweeney_ENZAMET_Lancet Onc'23.pdf"
# python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT00309985_Sweeney_CHAARTED_NEJM'15.pdf"
# python experiment-scripts/baseline_landing_ai_new.py --pdf_name "NCT01957436_Fizazi_PEACE1_Lancet'22.pdf"


# python experiment-scripts/baseline_file_search.py --pdf "dataset/NCT01809691_Aggarwal_SWOG1216_JCO'22.pdf" --provider gemini --model gemini-2.5-flash --workers 10
# python experiment-scripts/baseline_file_search.py --pdf "dataset/NCT02799602_Smith_ARASENS_NEJM'22.pdf" --provider gemini --model gemini-2.5-flash --workers 10
# python experiment-scripts/baseline_file_search.py --pdf "dataset/NCT02446405_Sweeney_ENZAMET_Lancet Onc'23.pdf" --provider gemini --model gemini-2.5-flash --workers 10
# python experiment-scripts/baseline_file_search.py --pdf "dataset/NCT00309985_Sweeney_CHAARTED_NEJM'15.pdf" --provider gemini --model gemini-2.5-flash --workers 10
# python experiment-scripts/baseline_file_search.py --pdf "dataset/NCT01957436_Fizazi_PEACE1_Lancet'22.pdf" --provider gemini --model gemini-2.5-flash --workers 10


# python experiment-scripts/baseline_file_search.py --pdf "dataset/NCT01809691_Aggarwal_SWOG1216_JCO'22.pdf" --provider gemini --model gemini-2.0-flash --workers 10
# python experiment-scripts/baseline_file_search.py --pdf "dataset/NCT02799602_Smith_ARASENS_NEJM'22.pdf" --provider gemini --model gemini-2.0-flash --workers 10
# python experiment-scripts/baseline_file_search.py --pdf "dataset/NCT02446405_Sweeney_ENZAMET_Lancet Onc'23.pdf" --provider gemini --model gemini-2.0-flash --workers 10
# python experiment-scripts/baseline_file_search.py --pdf "dataset/NCT00309985_Sweeney_CHAARTED_NEJM'15.pdf" --provider gemini --model gemini-2.0-flash --workers 10
# python experiment-scripts/baseline_file_search.py --pdf "dataset/NCT01957436_Fizazi_PEACE1_Lancet'22.pdf" --provider gemini --model gemini-2.0-flash --workers 10

# Native Gemini (JSON schema) data extraction – run from repo root with GEMINI_API_KEY set
python experiment-scripts/baseline_file_search_gemini_native.py --pdf "dataset/NCT00104715_Gravis_GETUG_EU'15.pdf" --model gemini-2.5-flash --workers 10
python experiment-scripts/baseline_file_search_gemini_native.py --pdf "dataset/NCT00268476_James_STAMPEDE_IJC'22.pdf" --model gemini-2.5-flash --workers 10
python experiment-scripts/baseline_file_search_gemini_native.py --pdf "dataset/NCT00309985_Kriayako_CHAARTED_JCO'18.pdf" --model gemini-2.5-flash --workers 10
python experiment-scripts/baseline_file_search_gemini_native.py --pdf "dataset/NCT00309985_Sweeney_CHAARTED_NEJM'15.pdf" --model gemini-2.5-flash --workers 10
python experiment-scripts/baseline_file_search_gemini_native.py --pdf "dataset/NCT01809691_Aggarwal_SWOG1216_JCO'22.pdf" --model gemini-2.5-flash --workers 10
python experiment-scripts/baseline_file_search_gemini_native.py --pdf "dataset/NCT01957436_Fizazi_PEACE1_Lancet'22.pdf" --model gemini-2.5-flash --workers 10
python experiment-scripts/baseline_file_search_gemini_native.py --pdf "dataset/NCT02446405_Sweeney_ENZAMET_Lancet Onc'23.pdf" --model gemini-2.5-flash --workers 10
python experiment-scripts/baseline_file_search_gemini_native.py --pdf "dataset/NCT02799602_Hussain_ARASENS_JCO'23.pdf" --model gemini-2.5-flash --workers 10
python experiment-scripts/baseline_file_search_gemini_native.py --pdf "dataset/NCT02799602_Smith_ARASENS_NEJM'22.pdf" --model gemini-2.5-flash --workers 10
