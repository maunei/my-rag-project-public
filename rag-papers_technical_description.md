# PAPERS-RAG: TECHNICAL SYSTEM EXPOSITION

## Purpose and General Workflow of the Application

Researchers often accumulate hundreds or even thousands of scientific articles in PDF format over many years of work. Eventually, a point is reached where the challenge is no longer obtaining papers, but instead being able to efficiently identify which articles are truly relevant to a particular biological question, computational problem, experimental method, or scientific hypothesis that one wishes to study in depth.

This application was designed to help navigate and interact with large local repositories of scientific PDFs using a combination of semantic retrieval, keyword-based searches, Boolean logic, metadata extraction, and AI-assisted contextual exploration.

The system allows users to perform semantic searches across the content of their local PDF collection, meaning that papers can be retrieved not only through exact word matches, but also through conceptual similarity. This can be combined with traditional keyword searches and Boolean operators such as AND, OR, and NOT in order to progressively refine and narrow down the results.

The searchable vector database is created locally from the PDFs and can be easily updated directly within the application whenever new papers are added to the repository directories. This allows the indexed collection to remain synchronized with the user’s evolving local scientific archive without requiring complex manual reconfiguration steps.

The search results are directly hyperlinked to the original PDF files, allowing users to immediately open the corresponding papers in separate browser tabs for manual inspection, reading, or verification.

In addition to the automated retrieval process, the application also allows the user to manually provide exact PDF base filenames through a dedicated text box. The metadata associated with those papers can then be merged with the metadata and retrieved chunks coming from the current search results. In practice, this makes it possible to force the inclusion of specific known reference papers into the contextual information sent to the AI system, even if those papers were not part of the original retrieval set.

To support these workflows, the application maintains an independent metadata layer for all indexed papers. This metadata is generated locally as `.json` files by a dedicated extraction pipeline that parses the PDF documents. Optionally, this locally extracted metadata can be enriched programmatically by querying NCBI/PubMed in order to retrieve additional curated publication metadata such as official titles, abstracts, publication dates, PMIDs, author lists, and DOI validation.

Once relevant papers have been identified, the user may choose between different forms of interaction with the retrieved information.

One possibility is to have a direct AI-assisted conversation within the application itself using the retrieved metadata and text chunks as contextual grounding information. Another possibility is to export that same contextual information into a standalone text file that can later be used with the user’s own preferred AI system, whether free or paid.

The application was intentionally designed in a way that does not force users into a single AI ecosystem. Instead, it allows the contextual information generated during retrieval to remain portable and reusable across different AI platforms.

For more detailed analysis workflows, the user can send selected papers into a second interface dedicated to deeper document-level interaction. In this mode, the selected PDFs may be uploaded to Gemini models hosted on Google Cloud Vertex AI, allowing the AI system to interact with entire documents rather than only with previously retrieved text excerpts.

To simplify this process, when papers are transferred into this deeper analysis workflow, the application automatically creates links to the corresponding PDF files — even when those files are deeply nested within complex directory structures. These links are organized into a dedicated folder so that the user can quickly locate, inspect, or manually upload the selected PDFs into their own preferred AI environment if desired.

Overall, the application attempts to transform a large and potentially unmanageable local PDF archive into a searchable, navigable, AI-assisted scientific research environment where semantic retrieval, metadata enrichment, contextual export, and conversational analysis can all work together within a unified workflow.

---

## SYSTEM ARCHITECTURE AND CORE PHILOSOPHY

PAPERS-RAG is a modular system designed to bridge local scientific PDF repositories with conversational AI systems and cloud-based large language models.

Unlike minimal Retrieval-Augmented Generation (RAG) prototypes, the system intentionally separates several independent but interconnected layers:

- local PDF storage
- semantic vector retrieval
- metadata extraction
- contextual export
- conversational AI interaction
- and optional cloud-scale document analysis

This architecture allows the application to remain flexible, reproducible, and largely independent from any single AI provider or indexing platform.

At its core, the system relies on two parallel infrastructures:

1) a vector retrieval layer built from the full text extracted from PDFs

2) an independent metadata layer containing structured bibliographic information stored as JSON files

These two layers work together during retrieval and AI interactions but remain logically decoupled. This separation makes it possible to:

- enrich metadata independently
- update bibliographic information without rebuilding embeddings
- maintain lightweight vector indexes
- preserve traceability of extracted information
- and support modular workflows

---

## 1. VECTOR SEARCH INFRASTRUCTURE (CHROMADB)

The semantic retrieval engine is built on top of a persistent ChromaDB database stored locally on disk.

The vector index is generated directly from the textual content extracted from scientific PDF documents rather than from external APIs or manually curated corpora.

The system uses the:

`BAAI/bge-small-en-v1.5`

embedding model through the fastembed library.

This embedding model was selected because it provides:

- strong semantic retrieval performance
- efficient CPU-based inference
- lightweight ONNX execution
- scalability to large local repositories

During indexing, the application extracts text from PDF pages and divides the content into overlapping text chunks.

Chunking is necessary because:

- scientific papers are often very large
- embedding models perform better on moderate text windows
- retrieval precision improves with localized segments
- overlapping windows help preserve continuity of context

Each chunk is associated with metadata such as:

- source file path
- inferred paper title
- chunk identifiers
- page ranges
- retrieval information

The resulting embeddings are stored in the ChromaDB collection and can later be queried through semantic similarity searches.

---

## 2. HYBRID SEMANTIC + BOOLEAN RETRIEVAL

A major design objective of PAPERS-RAG was to combine semantic retrieval with explicit Boolean logic.

The system therefore supports:

- semantic similarity searches
- exact keyword searches
- AND / OR / NOT operators
- grouped Boolean clauses
- hybrid retrieval strategies

This hybrid approach is important in scientific literature exploration because:

- some concepts are semantic
- some identifiers require exact matching
- gene symbols are literal
- acronyms are sensitive to spelling
- Boolean exclusion is often necessary

Users may progressively refine retrieval results using combinations of semantic and literal constraints.

For example:

- semantic retrieval can identify conceptually related papers
- keyword filters can enforce exact biological terms
- Boolean exclusions can remove irrelevant subtopics

This allows retrieval workflows that are considerably more flexible than conventional search systems.

---

## 3. INDEPENDENT METADATA PIPELINE

Parallel to the vector database, the application maintains an independent metadata infrastructure.

Each indexed PDF has a corresponding JSON file stored separately from the semantic index.

These metadata records may contain:

- titles
- inferred titles
- DOI candidates
- extracted abstracts
- PubMed abstracts
- PMIDs
- publication dates
- author lists
- enrichment provenance
- schema version information

Metadata extraction is performed locally through dedicated parsing scripts operating directly on the PDFs.
Optionally, the metadata can be enriched programmatically by querying PubMed through the E-utilities/Entrez infrastructure.

This enrichment pipeline may retrieve:

- official publication titles
- curated abstracts
- DOI validation
- publication metadata
- PMIDs
- author lists

This architecture allows the application to combine:

- locally extracted information
with
- authoritative biomedical metadata

while keeping the retrieval infrastructure independent from the enrichment process.

---

## 4. SCRIPT RESPONSIBILITIES AND MODULAR DESIGN

The application is composed of several specialized scripts that interact in a modular way.

**papers_paths.py**

Centralizes the repository paths and provides utilities for recursively identifying PDF files throughout the local document collection.

**indexer.py**

Responsible for:

- text extraction
- chunking
- embedding generation
- ChromaDB indexing
- semantic retrieval
- keyword retrieval
- Boolean aggregation
- index statistics

**extract_abstracts.py**

Traverses the PDF repository and creates the JSON metadata files associated with each paper.

**abstract_extraction.py**

Contains the lower-level extraction logic responsible for:

- title inference
- DOI detection
- abstract extraction
- metadata handling
- JSON path utilities
- metadata integration

**ncbi_pubmed.py**

Handles communication with the NCBI / PubMed infrastructure through Entrez API queries.

**rag_engine.py**

Acts as the orchestration layer between:

- retrieval
- contextual assembly
- AI interactions
- cloud uploads
- and conversational workflows

**pdf_server.py**

Implements a lightweight local HTTP service allowing PDFs to be opened directly through hyperlinks in the user interface.

**app.py**

The main Streamlit application coordinating:

- the graphical interface
- session states
- indexing controls
- retrieval logic
- export workflows
- AI interactions
- and cloud integration

---

## 5. USER INTERFACE AND WORKFLOW ORGANIZATION

The graphical interface is implemented using Streamlit and organized around a sidebar and two main operational tabs.

### SIDEBAR AND INDEX MANAGEMENT

The sidebar provides:

- indexing controls
- index statistics
- repository information
- synchronization tools
- and database status indicators

Users can trigger:

- creation of the vector database
- incremental updates
- repository synchronization
- metadata refresh operations

This allows the semantic index to remain synchronized with the evolving PDF repository.

### TAB 1: CONTEXTUAL RETRIEVAL AND QUICK CHAT

The first tab is designed for:

- semantic retrieval
- Boolean searches
- contextual exploration
- metadata inspection
- and rapid AI-assisted interactions

The interface supports:

- semantic search clauses
- keyword clauses
- grouped Boolean logic
- similarity thresholds
- manual paper inclusion
- contextual exports

Users may interact with an AI system grounded by:

- retrieved text chunks
- metadata
- abstracts
- selected contextual excerpts

The application can also export the generated context into standalone text files for use with external AI systems.

### TAB 2: DEEP DOCUMENT ANALYSIS

The second tab is dedicated to deeper full-document analysis workflows.

Selected papers from Tab 1 can be transferred into this environment for more extensive interaction.

When papers are staged for deeper analysis:

- symbolic links (or copies) are automatically created
- selected files are organized into dedicated folders
- upload workflows become simplified
- file discovery becomes easier

Users may then upload the full PDFs into Gemini models hosted through Google Cloud Vertex AI or upload the files manually to their preferred AI tool.

In this mode the AI system can interact with:

- entire papers
- methods sections
- larger scientific narratives
- extended contextual information
- and complete document structures

---

## 6. CONTEXT EXPORT AND AI INTEROPERABILITY

A major design principle of PAPERS-RAG is interoperability.

The application does not force users into a single AI platform.

Instead, it allows retrieval results and contextual information to remain portable.

Users may export:

- metadata
- abstracts
- retrieved chunks
- contextual summaries
- and selected references

into standalone text files that can later be used with:

- ChatGPT
- Claude
- Gemini
- local LLM systems
- or any other AI environment

This provides:

- portability
- reproducibility
- model comparison
- workflow flexibility
- and long-term contextual archiving

---

## 7. LOCAL PDF SERVING AND VERIFICATION

The application includes a lightweight local PDF server.

This allows:

- direct PDF hyperlinks
- browser-based rendering
- side-by-side reading
- evidence verification during AI conversations

The local server makes it possible to rapidly inspect source material while simultaneously interacting with the AI system.

This is particularly important in scientific workflows where retrieved claims often need immediate verification against the original publication.

---

## 8. CLOUD INTEGRATION

The system integrates with Gemini models hosted on Google Cloud Vertex AI.

The cloud integration layer supports:

- Google Cloud Storage uploads
- Gemini conversational APIs
- Vertex AI workflows
- authenticated cloud interactions
- and full-document AI analysis

Importantly, this cloud integration remains optional.

The core infrastructure:

- indexing
- retrieval
- metadata extraction
- PDF serving
- contextual export
- and local exploration

can all operate entirely on local infrastructure without requiring cloud services.

---

## OVERALL OBJECTIVE

The overall objective of PAPERS-RAG is to transform large collections of scientific PDFs into an interactive AI-assisted research environment.

The system combines:

- semantic retrieval
- keyword search
- Boolean filtering
- metadata enrichment
- contextual export
- conversational AI
- and deep document analysis

within a unified modular workflow specifically designed for scientific literature exploration and advanced research assistance.
