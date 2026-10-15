
import { useCallback, useEffect, useState } from "react";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL ??
  (import.meta.env.PROD ? "" : "http://127.0.0.1:8000");

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);

  const [loadingDocs, setLoadingDocs] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);

  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const loadDocuments = useCallback(async () => {
    setLoadingDocs(true);

    try {
      const response = await fetch(`${API_URL}/documents`);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Could not load documents.");
      }

      setDocuments(data.documents || []);
    } catch (err) {
      setError(
        err.message ||
          "Could not connect to the DocuMind AI backend."
      );
    } finally {
      setLoadingDocs(false);
    }
  }, []);

  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  async function handleUpload(event) {
    event.preventDefault();
    setError("");
    setNotice("");
    setResult(null);

    if (!selectedFile) {
      setError("Please choose a PDF or DOCX file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    setUploading(true);

    try {
      const response = await fetch(`${API_URL}/documents`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Document upload failed.");
      }

      setNotice(
        data.already_exists
          ? `${data.filename} is already in your library.`
          : `${data.filename} was saved and indexed successfully.`
      );

      setSelectedFile(null);

      const input = document.getElementById("library-file");
      if (input) input.value = "";

      await loadDocuments();
    } catch (err) {
      setError(err.message || "Could not upload the document.");
    } finally {
      setUploading(false);
    }
  }

  async function handleAsk(event) {
    event.preventDefault();
    setError("");
    setNotice("");
    setResult(null);

    if (documents.length === 0) {
      setError("Upload a document before asking a question.");
      return;
    }

    if (!question.trim()) {
      setError("Please enter a question.");
      return;
    }

    const formData = new FormData();
    formData.append("question", question.trim());

    setAsking(true);

    try {
      const response = await fetch(`${API_URL}/knowledge/ask`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Question answering failed.");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Could not retrieve an answer.");
    } finally {
      setAsking(false);
    }
  }

  async function handleDelete(documentId, filename) {
    const confirmed = window.confirm(
      `Remove "${filename}" and its indexed chunks from your library?`
    );

    if (!confirmed) return;

    setError("");
    setNotice("");
    setResult(null);

    try {
      const response = await fetch(
        `${API_URL}/documents/${documentId}`,
        { method: "DELETE" }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Could not delete the document.");
      }

      setNotice(`${filename} was removed from the library.`);
      await loadDocuments();
    } catch (err) {
      setError(err.message || "Could not delete the document.");
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="DocuMind AI home">
          <span className="brand-icon">D</span>
          <span>
            <span className="brand-name">DocuMind</span>
            <span className="brand-ai"> AI</span>
            <small>DOCUMENT INTELLIGENCE</small>
          </span>
        </a>

        <div className="topbar-status">
          <span className="status-dot" />
          Local AI workspace
        </div>
      </header>

      <main className="main-content">
        <section className="hero">
          <span className="eyebrow">
            YOUR PERSONAL KNOWLEDGE ENGINE
          </span>
          <h1>
            Your knowledge,
            <br />
            <span>connected.</span>
          </h1>
          <p>
            Build a reusable document library, ask questions across
            your saved knowledge, and inspect the evidence behind every answer.
          </p>
        </section>

        <section className="workspace">
          <div className="panel input-panel">
            <div className="panel-heading">
              <span className="step-number">01</span>
              <div>
                <h2>Knowledge library</h2>
                <p>Upload once and reuse your documents.</p>
              </div>
            </div>

            <form onSubmit={handleUpload}>
              <label className="field-label" htmlFor="library-file">
                ADD A DOCUMENT
              </label>

              <label className="upload-box" htmlFor="library-file">
                <span className="upload-icon">↑</span>
                <strong>
                  {selectedFile ? "Document selected" : "Choose a document"}
                </strong>
                <span>
                  {selectedFile
                    ? selectedFile.name
                    : "Select a PDF or Word document"}
                </span>
                <small>PDF or DOCX · Maximum 10 MB</small>
              </label>

              <input
                id="library-file"
                className="file-input"
                type="file"
                accept=".pdf,.docx"
                onChange={(event) =>
                  setSelectedFile(event.target.files?.[0] || null)
                }
              />

              <button
                className="submit-button"
                type="submit"
                disabled={uploading}
              >
                {uploading
                  ? "Saving and indexing..."
                  : "Add to knowledge base →"}
              </button>
            </form>

            <div className="library-heading">
              <div>
                <h3>Saved documents</h3>
                <p>{documents.length} document(s) in your library</p>
              </div>

              <button
                className="refresh-button"
                type="button"
                onClick={loadDocuments}
                disabled={loadingDocs}
              >
                {loadingDocs ? "Loading..." : "Refresh"}
              </button>
            </div>

            {loadingDocs && (
              <p className="library-empty">Loading saved documents...</p>
            )}

            {!loadingDocs && documents.length === 0 && (
              <p className="library-empty">
                Your library is empty. Upload a document to get started.
              </p>
            )}

            {!loadingDocs && documents.length > 0 && (
              <div className="library-list">
                {documents.map((doc) => (
                  <article className="document-row" key={doc.id}>
                    <div className="document-file-icon">
                      {doc.filename.toLowerCase().endsWith(".pdf")
                        ? "PDF"
                        : "DOC"}
                    </div>

                    <div className="document-details">
                      <strong className="document-name">
                        {doc.filename}
                      </strong>
                      <span>
                        {doc.chunk_count} indexed chunk(s)
                      </span>
                    </div>

                    <button
                      className="delete-button"
                      type="button"
                      aria-label={`Delete ${doc.filename}`}
                      title={`Delete ${doc.filename}`}
                      onClick={() => handleDelete(doc.id, doc.filename)}
                    >
                      Remove
                    </button>
                  </article>
                ))}
              </div>
            )}

            <div className="question-section">
              <div className="panel-heading question-heading">
                <span className="step-number">02</span>
                <div>
                  <h2>Ask your knowledge base</h2>
                  <p>Search across all indexed documents.</p>
                </div>
              </div>

              <form onSubmit={handleAsk}>
                <label className="field-label" htmlFor="question">
                  YOUR QUESTION
                </label>

                <textarea
                  id="question"
                  className="question-input"
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="e.g. How long is the refund period?"
                  rows={4}
                  maxLength={1000}
                />

                <button
                  className="submit-button"
                  type="submit"
                  disabled={asking || documents.length === 0}
                >
                  {asking ? "Searching your knowledge..." : "Ask DocuMind →"}
                </button>
              </form>
            </div>

            {notice && (
              <div className="success-message" role="status">
                {notice}
              </div>
            )}

            {error && (
              <div className="error-message" role="alert">
                {error}
              </div>
            )}

            <p className="privacy-note">
              <span aria-hidden="true">◈</span>
              This prototype stores document text and index data in a local
              SQLite database. Use only documents you are permitted to process.
            </p>
          </div>

          <div className="panel answer-panel">
            <div className="panel-heading">
              <span className="step-number">03</span>
              <div>
                <h2>Answer & evidence</h2>
                <p>Inspect the retrieved passages behind the answer.</p>
              </div>
            </div>

            {!result && !asking && (
              <div className="empty-state">
                <div className="empty-icon">✳</div>
                <h3>Your research starts here</h3>
                <p>
                  Add documents to your knowledge library, then ask a
                  question to see a grounded answer with its source passages.
                </p>
                <div className="empty-hint">
                  <span>01</span> Add documents to your library
                  <br />
                  <span>02</span> Ask across saved knowledge
                  <br />
                  <span>03</span> Review the evidence
                </div>
              </div>
            )}

            {asking && (
              <div className="loading-state" aria-live="polite">
                <span className="large-spinner" />
                <h3>Searching your knowledge</h3>
                <p>
                  Retrieving stored embeddings and generating a locally
                  powered answer. This may take a little time.
                </p>
              </div>
            )}

            {result && !asking && (
              <div className="result-content" aria-live="polite">
                <div className="answer-label">
                  <span className="answer-check">✓</span>
                  GENERATED ANSWER
                </div>

                <div className="answer-box">
                  <p>{result.answer}</p>
                </div>

                <div className="sources-heading">
                  <div>
                    <h3>Source passages</h3>
                    <p>Retrieved from your saved documents</p>
                  </div>
                  <span className="source-count">
                    {result.sources?.length || 0} found
                  </span>
                </div>

                {result.sources?.map((source, index) => (
                  <article
                    className="source-card"
                    key={`${source.document_id}-${source.chunk_index}-${index}`}
                  >
                    <div className="source-card-top">
                      <span>{source.filename || `Source ${index + 1}`}</span>
                      <span className="similarity-score">
                        Similarity: {Number(source.similarity).toFixed(3)}
                      </span>
                    </div>
                    <p>{source.text}</p>
                    <small className="source-location">
                      Chunk {source.chunk_index + 1}
                    </small>
                  </article>
                ))}

                <p className="verification-note">
                  Verify generated answers against their sources.
                  Similarity scores are not confidence percentages.
                </p>
              </div>
            )}
          </div>
        </section>

        <footer className="page-footer">
          <span>DocuMind AI</span>
          <span>React · FastAPI · Sentence Transformers · SQLite</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
