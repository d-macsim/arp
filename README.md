# Citibank APR Example

Small example project for running a DeepSeek-powered financial document analysis flow.

## Structure

```text
.
|- docs/
|  |- netflix/
|  `- overview_document.pdf
|- manifests/
|  `- netflix_reports.json
|- outputs/
|- prompts/
|  `- llm_analysis_prompt_template.md
|- .env
|- .gitignore
|- example_runthrough.py
|- report_pipeline.py
|- README.md
|- run_reports.py
`- requirements.txt
```

## Local setup

```powershell
C:\Users\bencr\OneDrive\Personal\Computing\Business Analytics\Citibank APR\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
C:\Users\bencr\OneDrive\Personal\Computing\Business Analytics\Citibank APR\.venv\Scripts\python.exe example_runthrough.py
```

If `python` is available on your PATH, the shorter command still works:

```powershell
python example_runthrough.py
```

## Batch Run

Dry-run the Netflix batch without spending API tokens:

```powershell
C:\Users\bencr\OneDrive\Personal\Computing\Business Analytics\Citibank APR\.venv\Scripts\python.exe run_reports.py --issuer netflix --dry-run
```

Run the 4 Netflix reports for real:

```powershell
C:\Users\bencr\OneDrive\Personal\Computing\Business Analytics\Citibank APR\.venv\Scripts\python.exe run_reports.py --issuer netflix
```

## Notes

- `.env` should contain `DEEPSEEK_API_KEY=...`
- `example_runthrough.py` loads the system prompt and user template from `prompts/llm_analysis_prompt_template.md`
- `run_reports.py` reads the Netflix input set from `manifests/netflix_reports.json`
- PDF text extraction is local and uses `pypdf` with `pdfplumber` fallback
- The batch runner writes extracted text, latest results, logs, and summary files into `outputs/<issuer>/`
- Each batch run is also archived under `outputs/<issuer>/runs/<run_id>/` so later runs do not overwrite prior summaries and result files
- The bundled example transcript inside `example_runthrough.py` is fictional
