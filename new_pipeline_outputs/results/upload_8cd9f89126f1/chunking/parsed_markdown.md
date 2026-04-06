<a id='06d89d2b-f17c-4a23-a564-20fb5fa8ad9e'></a>

EviSearch: A Human in the Loop System for Extracting and Auditing
Clinical Evidence for Systematic Reviews

<a id='13139a70-c618-47c5-8177-2d0887df1d1f'></a>

¹Naman Ahuja ¹Saniya Mulla ²Muhammad Ali Khan ²Zaryab Bin Riaz
²Kaneez Zahra Rubab Khakwani ²Mohamad Bassam Sonbol ²Irbaz Bin Riaz ¹Vivek Gupta
Arizona State University¹ Mayo Clinic²
riaz.irbaz@mayo.edu, vgupt140@asu.edu

<a id='29909fe2-3578-4cf9-bd18-8db6ab864d36'></a>

# Abstract

We present **EviSearch**, a multi-agent extraction system that automates the creation of ontology-aligned clinical evidence tables directly from native trial PDFs while guaranteeing per-cell provenance for audit and human verification. EviSearch pairs a PDF-query agent (which preserves rendered layout and figures) with a retrieval-guided search agent and a reconciliation module that forces page-level verification when agents disagree. The pipeline is designed for high-precision extraction across multimodal evidence sources (text, tables, figures) and for generating reviewer-actionable provenance that clinicians can inspect and correct. On a clinician-curated benchmark of oncology trial papers, EviSearch substantially improves extraction accuracy relative to strong parsed-text baselines while providing comprehensive attribution coverage. By logging reconciler decisions and reviewer edits, the system produces structured preference and supervision signals that bootstrap iterative model improvement. EviSearch is intended to accelerate living systematic review workflows, reduce manual curation burden, and provide a safe, auditable path for integrating LLM-based extraction into evidence synthesis pipelines.

<a id='58f1ab91-db77-4d48-a2e8-6540993b633b'></a>

# 1 Introduction
Structured extraction of clinical evidence from trial publications is a foundational step in evidence synthesis, meta-analysis, and clinical guideline development. Living systematic review platforms¹ curate interactive tables summarizing trial identifiers, treatment arms, patient characteristics, endpoints, and subgroup outcomes, enabling clinicians to compare therapies at a glance. However, constructing and maintaining these structured evidence tables remains a largely manual process requiring expert review of full-text PDFs. Recent advances in large language models (LLMs) have demonstrated strong

<a id='eae927f8-d154-480c-8126-220855acf48b'></a>

performance across medical reasoning, documentation, and summarization tasks (Zhou et al., 2023; Wang et al., 2025). Clinical evaluations and systematic reviews highlight the potential of LLMs to support diagnostic reasoning and information synthesis, while also revealing substantial variability and reliability concerns in real-world settings (Omar et al., 2024; Meng et al., 2024).

<a id='488dbdfb-d832-4874-b43a-a609ad4d0a8e'></a>

In the context of clinical trial extraction, hal-
lucinated or misattributed values can directly im-
pact downstream meta-analyses and treatment de-
cisions. Beyond reliability, clinical trial PDFs
present intrinsic technical challenges. Evidence
is distributed across heterogeneous modalities: nar-
rative text, complex tables, Kaplan–Meier plots,
and figure captions. Some schema fields require
global document-level reasoning (e.g., determining
whether quality-of-life outcomes were reported),
while others demand fine-grained cell-level extrac-
tion from structured tables or graphical figures.
Subgroup reporting (e.g., high- vs. low-volume
disease) introduces scope and normalization chal-
lenges that exceed simple pattern matching.

<a id='6e9ac669-eb7b-4828-a085-9a48dce96eaa'></a>

We present **EviSearch**, a multi-agent extraction framework designed to automate ontology-aligned evidence tables directly from native clinical trial PDFs while enforcing per-cell provenance and verification. EviSearch combines a direct PDF query agent that preserves multi-modal layout with a retrieval-guided search agent targeting page-level evidence. A reconciliation module adjudicates disagreements via forced page-level verification, and a human-on-the-loop interface exposes grounded attribution for auditing and feedback. We evaluate EviSearch on a clinician-curated benchmark whose schema mirrors the fields used in living evidence platforms for metastatic castration-sensitive prostate cancer (mCSPC). Our results show substantial improvements over baselines while achieving comprehensive attribution coverage.

<a id='0c5e10c3-99f1-4d86-8a9f-339ec70fd6e6'></a>

We encourage readers to try it at the following

<a id='98bd13d2-3b1f-467c-9d37-dd38296570c7'></a>

¹ https://mcspc.living-evidence.com/

<a id='198dbb37-fdae-4177-a2df-2fe2815b7fb5'></a>

1

<!-- PAGE BREAK -->

<a id='c782d247-dd51-4fbc-a671-1c0bbd12adac'></a>

<::transcription of the content
: flowchart::>
Figure 1: EviSearch system architecture

This flowchart illustrates the EviSearch system architecture, divided into four main sections: User, AI Processing and Retrieval, Human Validation Interface, and Long term Data Storage.

**User Section:**
-   **User** (main box)
    -   A **PDF icon** points to **Define Target Schema/Value to extract**.
    -   **Define Target Schema/Value to extract** (box)
        -   Contains **Field 1** (box) and **Field 2** (box).
        -   Arrows from **Field 1** and **Field 2** point to the **AI Processing and Retrieval** section.

**AI Processing and Retrieval Section:**
-   **AI Processing and Retrieval** (main box)
    -   **Query Direct PDF(upload)** (box)
        -   An arrow points to **Document Chunking**.
        -   An arrow points to **LLM**.
    -   **LLM** (box)
        -   An arrow points to **Data Output**.
    -   **Data Output** (box)
        -   Contains **value**, **reasoning**, **attribution** (sub-boxes).
        -   An arrow points to **Reconciliation Agent**.
    -   **Search and Retrieval** (box)
        -   An arrow from **Define Target Schema/Value to extract** (User section) points to this box.
        -   Contains **Document Indexing** (box).
            -   An arrow points to **Retrieval Agent**.
        -   **Retrieval Agent** (box)
            -   An arrow points to a box containing **value**, **reasoning**, **attribution**.
        -   A box containing **value**, **reasoning**, **attribution** (sub-boxes)
            -   An arrow points to **Reconciliation Agent**.
    -   **Document Chunking** (box)
        -   An arrow points to **Reconciliation Agent**.
        -   An arrow labeled **Update Index** points to the **Human Validation Interface** section.
    -   **Reconciliation Agent** (box)
        -   An arrow points to **Reconciliation Module**.
        -   Arrows point to **View source pdf** and **Chatbot** (within Human Validation Interface).
    -   **Reconciliation Module** (box)
        -   Contains text: `{"field": "Trial Size", value: 302, page: 4, evidence:..., confidence: 0.93}`.
        -   An arrow points to **Feedback and Edit value** (within Human Validation Interface).

**Human Validation Interface Section:**
-   **Human Validation Interface** (main box)
    -   An arrow labeled **Update Index** comes from **Document Chunking** (AI Processing and Retrieval section).
    -   **View source pdf** (box)
    -   **Chatbot** (box)
    -   **Feedback and Edit value** (box)
        -   An arrow points to **Final Report (value, csv, json)**.
        -   An arrow points to **Knowledge Base** (within Long term Data Storage).
    -   **Final Report (value, csv, json)** (box)

**Long term Data Storage Section:**
-   **Long term Data Storage** (main box)
    -   Contains a **database icon** labeled **Knowledge Base** (box).
<::chart::>

<a id='7430cdb9-2762-4dd8-b2fc-699860869807'></a>

link:
* Demo:
  asu.github.io/EviSearch/
* Video

<a id='8fee27f1-3744-442d-a7d9-c3dc27179c67'></a>

https://coral-lab-

<a id='32a5ba12-7bc1-4ea3-9956-467ff53252a6'></a>

## 2 Related Work

Understanding scientific documents has been widely studied in natural language processing and document AI. Domain-adapted pretraining models such as SciBERT improve representation learning for scholarly text (Beltagy et al., 2019), while large-scale corpora like S2ORC enable structured modeling of scientific papers at scale (Lo et al., 2020). Document question answering benchmarks such as DocVQA evaluate models on visually rich PDFs requiring joint reasoning over text and layout (Mathew et al., 2021), and layout-aware architectures including LayoutLM encode spatial structure alongside textual content to improve document understanding (Xu et al., 2020). OCR-free transformers such as Donut further advance end-to-end document reasoning without explicit text extraction (Kim et al., 2022). In parallel, structured table reasoning has progressed through models such as TAPAS, TaBERT, and TURL, which incorporate tabular representations into transformer architectures for semantic parsing and question answering (Herzig et al., 2020; Yin et al., 2020; Deng et al., 2020). Chart and plot understanding benchmarks, including ChartQA and PlotQA, demonstrate the visual and logical reasoning required to interpret graphical data (Masry et al., 2022; Methani et al., 2020). Retrieval-augmented generation further grounds model outputs in external evidence to improve factual consistency (Lewis et al., 2020), and recent multimodal LLMs extend this paradigm to documents and images. However,

<a id='10f22136-3881-4463-bccf-3b04a78a65ef'></a>

existing approaches typically address isolated com-ponents of the problem, document layout modeling, table reasoning, chart interpretation, or retrieval-based grounding, rather than providing a unified system that performs schema-constrained extrac-tion across multimodal scientific PDFs with ex-plicit per-cell provenance and auditable verifica-tion. EviSearch brings these strands together into a single pipeline designed specifically for ontology-aligned clinical evidence synthesis.

<a id='9441d675-bfed-4f93-b23d-feb1a294627c'></a>

## 3 EviSearch System Architecture

EviSearch is a multi-stage, multi-agent extraction pipeline that fills a structured clinical evidence table from a trial publication PDF, while producing grounded, auditable attribution for every extracted value. The pipeline consists of four stages: (i) document parsing and chunking, (ii) parallel extraction by two independent agents, (iii) automated reconciliation, and (iv) human review and feedback through a web interface. Figure 1 provides an overview of the full architecture.

<a id='3fb76f2e-7dbb-4f7d-a9b5-4fba1b5199b7'></a>

## 3.1 Column Schema and Batching
The output schema consists of 133 columns drawn from the structured evidence tables used in the LISR living evidence synthesis platform ². Each column is paired with a natural-language definition that specifies the required value, reporting conventions, and fallback behavior (e.g., "use Not reported if missing"). Columns are annotated with an evaluation category (numeric_tolerance, structured_text, or exact_match) and grouped by clinical section: trial characteristics, population characteristics, efficacy outcomes, subgroup reporting, adverse events, and demographic breakdowns.

<a id='d7abc7df-98ee-4228-bccc-059d47174a31'></a>

²https://mcspc.living-evidence.com/

<a id='dfc4cf9c-aa4a-4e05-88c1-65641e098881'></a>

2

<!-- PAGE BREAK -->

<a id='d6c6ceb5-8015-44a1-b7e8-ef2b8e166f07'></a>

2. Column groups
Use all column groups or pick specific ones. Hover a badge to see its columns.
option Use all column groups: [ ]
Add a column group...
2 groups selected
Gleason score - N (%) x Median OS (mo) x

3. Extract
Run Agentic Extractor
Load existing

Extraction results
Loaded 18 queries Running 2 methods in parallel per batch.

<::table
| | Direct PDF | Search-Based | Total | Status |
|:---|:---|:---|:---|:---|
| | 8 populated 2 not reported | 0 populated 10 not reported | 10 | Done |
::>

View Attribution →

<::table
| Column | Direct PDF | Search-Based |
|:---|:---|:---|
| Gleason score - N (%) | ≤ 7 | Treatment What number and percentage of participants included in the treatment arm of the clinical trial with a Gleason score of 7 or less? If reported, include count and/or percentage. If multiple relevant arms are reported, list each arm separately. Use 'Not reported' if missing. | 122 (18.7%) | Not reported |
| Gleason score - N (%) | ≤ 7 | Control What number and percentage of patients with a Gleason score of 7 or less in the control arm of the trial? If reported, include count and/or percentage. If multiple relevant arms are reported, list each arm separately. Use 'Not reported' if missing. | 118 (18.0%) | Not reported |
| Gleason score - N (%) | ≥ 8 | Treatment What number and percentage of participants in the treatment arm of the clinical trial with a Gleason score of 8 or higher? If reported, include count and/or percentage. If multiple relevant arms are reported, list each arm separately. Use 'Not reported' if missing. | 505 (77.6%) | Not reported |
| Gleason score - N (%) | ≥ 8 | Control What number and percentage of patients in the control arm of the clinical trial with a Gleason score of 8 or higher? If reported, include count and/or percentage. If multiple relevant arms are reported, list each arm separately. Use 'Not reported' if missing. | 516 (78.9%) | Not reported |
::>

Figure 2: Extraction Interface

<a id='905832c3-fe42-44dc-96b7-8a66b7c71acc'></a>

To enable parallelism while maintaining inter-
column context, columns are batched using a group-
aware packing algorithm: columns within the same
clinical section are kept together, and batches are
limited to a maximum of 15 columns to balance
context length against reliability. Groups exceed-
ing this limit are split into sequential sub-batches;
smaller groups are merged greedily until the batch
limit is reached. Both extraction agents and the
reconciliation agent operate over the same batch
structure, enabling direct comparison at the column
level.

<a id='93d1e725-6acf-4ef1-80a6-e56641458bfe'></a>

## 3.2 Document Parsing and Retrieval Indexing

Each PDF is parsed using Landing AI's dpt-2-latest Document Parse model, which produces element-level chunks with associated page numbers and modality labels (text, table, figure). The output is stored as structured JSON and a rendered Markdown representation preserving table layout for downstream use.

<a id='bd4748b6-00cb-4012-b458-9b392fb8c2a1'></a>

For semantic retrieval, each chunk is embedded
using OpenAI's text-embedding-3-large model
(3,072 dimensions), producing a per-document in-
dex over page-level chunks. Embeddings are com-
puted in batches of 100 and indexed for cosine
similarity search. This index is used exclusively by
the Search Agent (§3.4) and is not shared with the
PDF Query Module.

<a id='96f8bb1e-cf6c-4acd-bd26-2483397859b6'></a>

Document
Select or upload a PDF, then prepare it for questions.

Upload PDF
CHOOSE FILE No fil...hosen

Prepare document Ready. You can ask questions.

Mode
option Quick — Direct PDF chat (faster, no attribution): [x]

Ask a question

What was the median age of patients in the treatment arm?

DIRECT PDF
63.0 (57.0-68.2)

SEARCH-BASED
63.0 years

RECONCILED
63.0 (57.0-68.2)

Reasoning: Source A provides a more complete value including the IQR, which is consistent with Source B's value.

View attribution Feedback

e.g. What was the treatment arm?

<::transcription of the content
: chart::>

Figure 3: Attribution Interface

<a id='b29344b1-af13-410f-b5c7-cbdc538627d6'></a>

## 3.3 PDF Query Module (Agent A)

The PDF Query Module submits the full PDF binary alongside each column batch to Gemini-2.5-Flash via the File API. This preserves the document's native structure including figures, multi-column layouts, and table formatting, without relying on parsed text. The model is prompted with the column definitions and instructed to return, for each column, a structured tuple of (value, reasoning, attribution: {page, modality, verbatim_quote}). Outputs are constrained via tool-based structured generation (function calling with a strict JSON schema), and temperature is set to 0 for deterministic extraction.

<a id='28ab8a64-c28b-4b8e-a685-c7096530aa44'></a>

This module is most effective on columns requir-
ing global document context, such as trial design,
eligibility criteria, and qualitative judgments Be-
cause the PDF is re-uploaded per API call, input
token counts include document tokens at each invo-
cation; this is reflected in the cost analysis (§5.3).

<a id='150a9b4a-4494-4360-9f87-5ece6ddb8646'></a>

## 3.4 Search Agent (Agent B)

The Search Agent operates over the same column batches using a tool-based agentic loop over the parsed document representation, targeting fine-grained evidence in tables, figures, and results sections. It is powered by Gemini-2.5-Flash at temperature 0, with access to three tools:

*   search_chunks: Performs semantic search over the document index using the column definition as the query, returning the top-k (k = 5) most relevant pages with full content.
*   get_chunks_by_page: Loads the full parsed

<a id='552516b0-063e-492d-b207-d8c6f3e8823f'></a>

3

<!-- PAGE BREAK -->

<a id='5850f7a7-8acd-4295-8b5f-44083af748e5'></a>

content of specified pages by number, en-
abling targeted follow-up.

<a id='e5b05a63-bf8d-4e8a-9c10-f973d5b80284'></a>

* submit_extraction: Submits the final extracted values with structured attribution once all columns are resolved.

<a id='b48a4e89-f149-481f-bac8-2e53c699d8d4'></a>

The agent follows an extract-first policy: it attempts extraction from the initial batch con-text before invoking retrieval tools for unresolved columns. A global deduplication mechanism tracks pages already provided to the agent within a ses-sion; subsequent requests for the same page re-turn a cache pointer rather than re-sending con-tent, bounding the effective context size and pre-venting redundant token expenditure. Outputs in-clude a (value, reasoning, attribution) tu-ple for each column, with attribution normalised to {page, modality} where modality ∈ {text, table, figure}.

<a id='7e9574bd-49d2-47ac-ae13-f0b32e541a40'></a>

## 3.5 Reconciliation Agent

The Reconciliation Agent receives the outputs of Agent A and Agent B for each column batch and adjudicates the final value. It applies a two-pass protocol:

<a id='631395b7-e07e-4275-a1f8-de3243feb4af'></a>

**Pass 1: Agreement detection (no tool use):**
Columns are resolved immediately if: (a) both
agents report *Not reported*; (b) both values are iden-
tical; or (c) both agents agree on page and modality
and one value is a strict superset of the other (e.g.,
a more complete numeric expression). These are
assigned both_correct.

<a id='906c1915-c17a-42d1-be6b-b8af47208594'></a>

**Pass 2: Verified resolution (forced tool use):**
In case of conflicts, where one agent reports
a value and the other reports *Not reported*, or
where values differ: require the agent to call
`get_page` before submitting. This tool returns
both the full parsed text and a rendered page
image for the disputed page, enabling multi-
modal verification. The agent then submits
a reconciled value with one of four verifica-
tion labels: both_correct, A_correct_B_wrong,
B_correct_A_wrong, or both_wrong. Outputs are
validated against a strict schema before being writ-
ten to disk.

<a id='924af1b2-e374-4968-960f-b7eed3d4b50d'></a>

Columns labelled both_wrong indicate that neither agent produced a verifiable answer; these are surfaced as low-confidence in the review interface for targeted human attention.

<a id='289d40d0-51d0-4e9d-8d80-3ad847773d53'></a>

## 3.6 Human Review, Auditability, and Verification Modes

EviSearch is designed around a _human-on-the-loop_ principle: the system operates fully autonomously by default, but every extraction decision is grounded in verifiable evidence that a human expert can inspect, challenge, or correct at any level of granularity. This is achieved through two complementary verification workflows.

<a id='a27691b0-838d-4ecf-b6d6-10a9e90c8997'></a>

Automated mode. For the common case, the reconciliation agent resolves all columns independently. Each extracted value is stored with explicit provenance; source page, modality, verbatim quote, and the reconciler's reasoning and rendered in the web interface with chunk-level highlighting directly in the source PDF. Clinicians receive a completed table in which every cell is one click away from its evidence.

<a id='8b388422-4379-4c78-8adf-0a329caf71cc'></a>

**Human-assisted mode.** For columns the system
flags as uncertain (both_wrong), or any column
a reviewer chooses to inspect, the same evidence
infrastructure used by the reconciliation agent is
available to the human directly. The reviewer sees
Agent A's answer, Agent B's answer, the recon-
ciler's judgment, and the attributed page content
side by side, and can accept one candidate or write
a corrected value. Critically, the human is never au-
diting a black box: they review the same grounded
evidence chain the reconciler produced.

<a id='3c982ef3-ff76-41bc-ac76-61f003c31374'></a>

# 4 Experiments

**Dataset:** We evaluate EviSearch on a benchmark of peer-reviewed clinical trial papers sampled from our clinician-curated dataset from the ontology of metastatic castration-sensitive prostate cancer (mC-SPC). For mCSPC, the clinical ontology encompasses standardized trial characteristics including trial identifiers, treatment and control arms, systemic therapy regimens, comparator groups, patient population summaries (e.g., age, performance status, disease volume), outcome measures (e.g., overall survival, progression-free survival), and subgroup results stratified by prognostic features such as high- vs. low-volume disease, synchronous vs. metachronous metastases. This mirrors the structured evidence presented in living systematic reviews of first-line mCSPC therapies, which summarize study design, patient characteristics, and comparative outcomes in tabular structures. Each paper in the benchmark is annotated with a structured schema of columns covering these clinical

<a id='87fca4e5-2150-4aba-9c41-78c5d6144bcd'></a>

4

<!-- PAGE BREAK -->

<a id='121d9d86-9451-4c29-b070-dcd071e9b0c5'></a>

<table id="4-1">
<tr><td id="4-2">Method</td><td id="4-3" colspan="3">Numeric Columns</td><td id="4-4" colspan="3">Free-Text Columns</td><td id="4-5" colspan="3">Overall (All)</td></tr>
<tr><td id="4-6"></td><td id="4-7">Corr.</td><td id="4-8">Comp.</td><td id="4-9">Ovrl.</td><td id="4-a">Corr.</td><td id="4-b">Comp.</td><td id="4-c">Ovrl.</td><td id="4-d">Corr.</td><td id="4-e">Comp.</td><td id="4-f">Ovrl.</td></tr>
<tr><td id="4-g">Gemini 2.5 Flash (PDF upload)</td><td id="4-h">83.5</td><td id="4-i">81.7</td><td id="4-j">82.6</td><td id="4-k">78.3</td><td id="4-l">76.7</td><td id="4-m">77.5</td><td id="4-n">82.5</td><td id="4-o">80.9</td><td id="4-p">81.7</td></tr>
<tr><td id="4-q">Gemini 2.5 Flash (parsed Doc)</td><td id="4-r">79.4</td><td id="4-s">77.4</td><td id="4-t">78.4</td><td id="4-u">72.0</td><td id="4-v">71.7</td><td id="4-w">71.8</td><td id="4-x">78.3</td><td id="4-y">76.7</td><td id="4-z">77.5</td></tr>
<tr><td id="4-A">GPT-4.1 (parsed Doc)</td><td id="4-B">84.7</td><td id="4-C">86.5</td><td id="4-D">85.6</td><td id="4-E">76.7</td><td id="4-F">78.0</td><td id="4-G">77.3</td><td id="4-H">83.3</td><td id="4-I">84.9</td><td id="4-J">84.1</td></tr>
<tr><td id="4-K">EviSearch (Ours)</td><td id="4-L">91.0</td><td id="4-M">92.3</td><td id="4-N">91.7</td><td id="4-O">89.7</td><td id="4-P">87.7</td><td id="4-Q">88.7</td><td id="4-R">90.9</td><td id="4-S">91.6</td><td id="4-T">91.3</td></tr>
</table>
Table 1: Extraction performance comparison across methods. **Corr.** = Correctness, **Comp.** = Completeness, **Ovrl.**
= Overall (mean of Corr. and Comp.). All scores are in %.

<a id='75f58977-4349-44bb-b728-be264bc7304c'></a>

<::bar chart: Overall extraction performance across evidence modalities.: Gemini 2.5 Flash (PDF) (blue), Gemini 2.5 Flash (Parsed) (red), GPT-4.1 (Parsed) (beige), EviSearch (Ours) (gray) with Overall Score (%) on the y-axis and categories Text, Table, Figure on the x-axis. Text category scores: Gemini 2.5 Flash (PDF) 81.6, Gemini 2.5 Flash (Parsed) 78.1, GPT-4.1 (Parsed) 87.3, EviSearch (Ours) 91.2. Table category scores: Gemini 2.5 Flash (PDF) 83.6, Gemini 2.5 Flash (Parsed) 78.2, GPT-4.1 (Parsed) 73.5, EviSearch (Ours) 91.7. Figure category scores: Gemini 2.5 Flash (PDF) 65.6, Gemini 2.5 Flash (Parsed) 51.6, GPT-4.1 (Parsed) 76.6, EviSearch (Ours) 86.7.::>Figure 4: Overall extraction performance across evidence modalities.

<a id='0f255fa0-0341-4a9e-8656-0183b55dd857'></a>

elements, and for every reported value, clinicians
provide gold-standard evidence attribution includ-
ing source page and modality (text, table, figure).

<a id='76b26c2e-f07d-42cb-b53f-a287bf170642'></a>

## 4.1 Baselines:
We compare EviSearch against three baselines. (1) Gemini 2.5 Flash (PDF upload), which uploads pdf and extracts structured outputs (2) Gemini 2.5 Flash (parsed Doc), which extracts values from Landing-AI-generated markdown text extracted from the PDF during pre-processing. (3) GPT-4.1 (parsed Doc), applied to the same parsed markdown representation.

<a id='45246804-2ab6-407b-9e5c-f333ba9e96fc'></a>

**Evaluation** For evaluation, all columns are di-vided into two categories, numerical and free text. We use an LLM judge with specifically crafted instructions for both categories to evaluate predic-tions against ground-truth outputs. We report three metrics: *Correctness*, the proportion of extracted values matching the gold standard; *Completeness*, the proportion of required schema columns suc-cessfully filled; and *Overall*, defined as the mean of correctness and completeness. For numeric fields, we apply tolerant matching to account for rounding

<a id='b467e8e2-dfb5-468a-9db7-10f458303349'></a>

and formatting differences.

<a id='f32eb631-e72b-482b-b645-5b2624105c36'></a>

## 5 Results and Analysis
### 5.1 Overall Extraction Performance
Table 1 reports extraction performance across all four methods on our benchmark. EviSearch achieves the highest scores across all metrics and both column types: 91.3% overall (90.9% correctness, 91.6% completeness), compared to the best baseline GPT-4.1 (parsed Doc) at 84.1%, a 7.2 point gain. This gap is meaningful in a clinical context, where missed or incorrect values can propagate into downstream meta-analyses.

<a id='808d11ef-d9b6-4342-9893-c9f36ffaca30'></a>

Among the baselines, GPT-4.1 attains 85.6% on
numeric columns but drops to 77.3% on free-text,
a 8.3 point gap suggesting that it relies more on
surface-level pattern matching than on genuine se-
mantic understanding of clinical language. Gem-
ini 2.5 Flash (parsed Doc) performs worst overall
(77.5%), confirming that feeding a model parsed
markdown, even with well-preserved structure, is
insufficient when evidence is spread across hetero-
geneous modalities. Notably, Gemini 2.5 Flash
improves by 4.2 points when given the native PDF

<a id='00614298-4058-4fa2-96aa-e978693a7e75'></a>

5

<!-- PAGE BREAK -->

<a id='4703fc4f-c7d2-46cc-9f56-554201264552'></a>

<table id="5-1">
<tr><td id="5-2">Method</td><td id="5-3">In Tok.</td><td id="5-4">Out Tok.</td><td id="5-5">Total Tok.</td><td id="5-6">API Calls</td></tr>
<tr><td id="5-7">Gemini (Native PDF)</td><td id="5-8">135,564</td><td id="5-9">13,930</td><td id="5-a">149,494</td><td id="5-b">39</td></tr>
<tr><td id="5-c">EviSearch</td><td id="5-d">603,350</td><td id="5-e">39,448</td><td id="5-f">642,798</td><td id="5-g">79</td></tr>
<tr><td id="5-h">Gemini (Parsed Text)</td><td id="5-i">957,761</td><td id="5-j">12,521</td><td id="5-k">970,282</td><td id="5-l">39</td></tr>
<tr><td id="5-m">GPT-4.1 (Parsed Text)</td><td id="5-n">990,819</td><td id="5-o">11,107</td><td id="5-p">1,001,926</td><td id="5-q">39</td></tr>
</table>
Table 2: Average token usage and API calls per docu-
ment.

<a id='dfc5a9b4-955d-4634-bd66-4d852e847a78'></a>

directly (81.7%), highlighting the value of preserv-
ing document layout and figures.

<a id='8f032bfd-cd83-4371-bde3-10ea2945d750'></a>

5.2 Performance by Evidence Source Modality
Of the 667 evidence fields in our benchmark, 53.4% originate from text passages, 41.8% from tables, and 4.8% from figures, meaning over 46% of the evidence cannot be recovered from plain text alone. This distribution reflects the real structure of clinical trial papers, where key outcomes are often reported in dense result tables or survival curves rather than narrative prose. Figure 4 breaks down performance by source modality as determined by gold-standard attribution.

<a id='9e45c6d9-d6d8-44a7-8642-2fa94bc2a866'></a>

The modality breakdown reveals where each system structurally struggles. All baselines degrade on figure-sourced evidence: Gemini 2.5 Flash (parsed Doc) drops sharply to 51.6%, a gap of 26.5 points behind EviSearch (86.7%) on the same category. This confirms that vanilla text-based prompting simply cannot recover values embedded in charts or plots. Table extraction further separates the methods, GPT-4.1 drops 13.8 points moving from text to table fields (87.3 -> 73.5), likely because clinical result tables involve complex multi-row, multi-column structures that require targeted retrieval rather than global document context. EviSearch, by contrast, maintains near-constant performance across all three modalities (91.2 -> 91.7 -> 86.7), a robustness that directly reflects the complementary strengths of its two agents: the PDF query module handles layout-sensitive and figure-rich content, while the search agent specialises in structured table retrieval.

<a id='e61b2ba8-d2dd-43ad-82d1-f87e1ea13278'></a>

## 5.3 API Cost and Efficiency

Table 2 shows average token usage per document. EviSearch uses 642,798 total tokens over 79 API calls, compared to 149,494 for Gemini native and ~1M tokens for the parsed baselines. The higher cost reflects the dual-agent and reconciliation ar-

<a id='d61e7c88-ca8a-4611-91ca-3b0f837d50b3'></a>

chitecture, as each column batch is processed in-
dependently by two extraction agents and subse-
quently verified through forced page-level adjudi-
cation when disagreements arise. In addition, per-
cell provenance storage and structured tool-based
generation introduce modest overhead relative to
single-pass prompting.

<a id='9bdadfee-76a4-4763-beb4-bce3ef7736c4'></a>

However, this additional computation is directly tied to auditability guarantees. Unlike parsed-text baselines that consume large token budgets without preserving grounded attribution, EviSearch converts its token expenditure into verifiable evidence chains: every extracted value is traceable to a specific page and modality, and disagreements trigger explicit multimodal verification rather than silent failure. Notably, EviSearch remains substantially more efficient than parsed-text pipelines, which exceed 970k-1M tokens per document due to repeated inclusion of full document context. The retrieval-guided Search Agent and deduplicated page caching constrain context growth, ensuring that additional cost scales with uncertainty rather than document length.

<a id='6503f520-d60d-4434-8743-611c6fba0104'></a>

From a deployment perspective, the cost pro-
file reflects a deliberate tradeoff: modestly higher
API usage than native PDF prompting in exchange
for significant gains in accuracy (+7.2 overall) and
near-complete attribution coverage. In clinical ev-
idence synthesis workflows, where extraction er-
rors can propagate into downstream meta-analyses,
this cost-accuracy tradeoff favors reliability and
auditability over minimal token usage.

<a id='bd14976b-ba1a-4e3b-a792-240f15ef3f3e'></a>

# 6 Conclusion

We introduced EviSearch, a human-on-the-loop system combining direct PDF querying, retrieval-guided extraction, and principled reconciliation to produce accurate, auditable clinical evidence tables from trial publications. EviSearch substantially outperforms strong baselines while providing near-complete provenance coverage, with every extracted value traceable to its source.

<a id='d391d4e2-c737-4111-9dfa-87e7e46bae41'></a>

Beyond accuracy, EviSearch acts as a data fly-
wheel: reviewer corrections and reconciliation de-
cisions generate structured supervision signals for
continual improvement. We envision it as a prac-
tical step toward trustworthy automation in living
systematic reviews, reducing manual burden while
preserving clinical oversight and reproducibility.

<a id='91ae0061-0267-4da5-b65b-05a7df458a2b'></a>

6

<!-- PAGE BREAK -->

<a id='a7cb3b3f-7270-48f7-a502-728a40a693aa'></a>

# Limitations

While EviSearch demonstrates strong extraction performance, several limitations remain. The system relies on LLMs whose outputs are probabilistic and may exhibit reasoning errors or misinterpretation of nuanced clinical terminology. Evaluation uses an LLM-based judge, which improves scalability but may not capture all clinically meaningful distinctions relative to full expert adjudication. Token costs are non-trivial due to the dual-agent architecture, potentially limiting large-scale deployment. Finally, full automation of clinical evidence synthesis is neither the goal nor currently advisable; responsible use requires expert oversight, particularly when outputs inform meta-analyses or clinical decisions.

<a id='dfa42ab4-a4e5-4db4-96c8-a00bfaa39975'></a>

# Ethics Statement
EviSearch processes publicly available, peer-reviewed trial PDFs that contain no personally identifiable patient information. The system stores explicit provenance for every extracted value to enable traceability and error correction; reviewer edits are logged as structured feedback but no private user data is retained. EviSearch relies on third-party LLM APIs (e.g., Gemini), and users must supply their own credentials. Software components are released under the Apache 2.0 License; responsible deployment requires adherence to applicable data governance policies and clinical oversight standards.

<a id='4ef187f3-0385-4c07-8872-ba9e7b2338b8'></a>

References
Iz Beltagy, Kyle Lo, and Arman Cohan. 2019. Scibert:
A pretrained language model for scientific text. In
Proceedings of EMNLP-IJCNLP.

Xiang Deng, Huan Sun, Alyssa Lees, You Wu, and
Cong Yu. 2020. Turl: Table understanding through
representation learning. In Proceedings of VLDB.

Jonathan Herzig, Paweł Krzysztof Nowak, Thomas
Müller, Francesco Piccinno, and Julian Martin Eisen-
schlos. 2020. Tapas: Weakly supervised table parsing
via pre-training. In Proceedings of ACL.

Geewook Kim, Teakgyu Hong, Moonbin Yim,
JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Won-
seok Hwang, Sangdoo Yun, Dongyoon Han, and Se-

<a id='4a9eeb54-4ed8-4a19-ba55-b2da03c48200'></a>

unghyun Park. 2022. Ocr-free document understanding transformer. In *Proceedings of the European Conference on Computer Vision (ECCV)*.

<a id='19fd5fe6-2982-40f3-a841-4d8dbd51524f'></a>

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. In *Proceedings of NeurIPS*.

<a id='d175627b-62cb-4893-b437-cabae8650dc4'></a>

Kyle Lo, Lucy Lu Wang, Mark Neumann, Rodney Kin-
ney, and Daniel Weld. 2020. S2orc: The semantic
scholar open research corpus. In *Proceedings of*
ACL.

<a id='4ed2b5c5-162c-4379-92bc-2a0424825550'></a>

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty,
and Enamul Hoque. 2022. Chartqa: A benchmark
for question answering about charts with visual and
logical reasoning. arXiv preprint arXiv:2203.10244.

<a id='fcc9afad-86bb-4681-885b-6adb9a622e8f'></a>

Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. 2021. Docvqa: A dataset for vqa on document images. In Proceedings of WACV.

<a id='d8f180ee-2030-414a-a5bc-6e603d1bc019'></a>

Xiangbin Meng, Xiangyu Yan, Kuo Zhang, and 1 others.
2024. The application of large language models in
medicine: A scoping review. *iScience*, 27(109713).

<a id='df619416-8b8b-482f-95b9-cb26df155cb3'></a>

Nitesh Methani, Pritha Ganguly, Mitesh M. Khapra,
and Pratyush Kumar. 2020. Plotqa: Reasoning over
scientific plots. In Proceedings of the IEEE/CVF Win-
ter Conference on Applications of Computer Vision
(WACV).

<a id='8873c88d-30ea-44a7-a94d-32f7c86823b9'></a>

Mahmud Omar, Girish N. Nadkarni, Eyal Klang, and
Benjamin S. Glicksberg. 2024. Large language mod-
els in medicine: A review of current clinical trials
across healthcare applications. PLOS Digital Health,
3(11 November):e0000662.

<a id='9fc9c771-0cec-446b-b7e0-2db3a05e4424'></a>

Wenxuan Wang and 1 others. 2025. A survey of llm-based agents in medicine: How far are we from bay-max? arXiv preprint arXiv:2502.11211.

<a id='098d5fa5-8e39-4992-ad6f-cc707cf22ad2'></a>

Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu
Wei, and Ming Zhou. 2020. Layoutlm: Pre-training
of text and layout for document image understanding.
In *Proceedings of KDD*.

<a id='ff496a52-4287-4b75-bbeb-773a1bffad8e'></a>

Pengcheng Yin, Graham Neubig, Wen-tau Yih, and Se-
bastian Riedel. 2020. Tabert: Pretraining for joint
understanding of textual and tabular data. In Proceed-
ings of the Association for Computational Linguistics
(ACL).

<a id='5cdb8fb6-6678-40e4-bbd1-55df48bc761b'></a>

Hongjian Zhou and 1 others. 2023. A survey of large
language models in medicine: Progress, application,
and challenge. arXiv preprint arXiv:2311.05112.

<a id='d6a5646f-f56b-456d-bfc3-63f3398a2716'></a>

7