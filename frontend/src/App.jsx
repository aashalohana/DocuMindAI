import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setResult(null);

    if (!file) {
      setError("Please select a PDF or Word document first.");
      return;
    }

    if (!question.trim()) {
      setError("Please enter a question about your document.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("question", question.trim());

    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        const detail = Array.isArray(data.detail)
          ? data.detail.map((item) => item.msg).join(", ")
          : data.detail;

        throw new Error(detail || "The request could not be completed.");
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message ||
          "Could not connect to DocuMind AI. Check whether the backend is running."
      );
    } finally {
      setLoading(false);
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
          <span className="eyebrow">YOUR PERSONAL DOCUMENT ASSISTANT</span>
          <h1>
            Your documents,
            <br />
            <span>finally understood.</span>
          </h1>
          <p>
            Upload a document, ask a question in your own words, and get an
            answer supported by relevant source text.
          </p>
        </section>

        <section className="workspace">
          <div className="panel input-panel">
            <div className="panel-heading">
              <span className="step-number">01</span>
              <div>
                <h2>Ask your document</h2>
                <p>Choose a file and enter your question.</p>
              </div>
            </div>

            <form onSubmit={handleSubmit}>
              <label className="field-label" htmlFor="document">
                YOUR DOCUMENT
              </label>

              <label className="upload-box" htmlFor="document">
                <span className="upload-icon">↑</span>
                <strong>
                  {file ? "Document selected" : "Choose a document"}
                </strong>
                <span>
                  {file
                    ? file.name
                    : "Click here to select a PDF or Word file"}
                </span>
                <small>PDF or DOCX · Maximum 10 MB</small>
              </label>

              <input
                id="document"
                className="file-input"
                type="file"
                accept=".pdf,.docx"
                onChange={(event) => {
                  setFile(event.target.files?.[0] || null);
                  setError("");
                  setResult(null);
                }}
              />

              <label className="field-label question-label" htmlFor="question">
                YOUR QUESTION
              </label>

              <textarea
                id="question"
                className="question-input"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="e.g. How long do customers have to request a refund?"
                rows={4}
                maxLength={1000}
              />

              <button className="submit-button" type="submit" disabled={loading}>
                {loading ? (
                  <>
                    <span className="spinner" />
                    Finding your answer...
                  </>
                ) : (
                  <>
                    Ask DocuMind <span aria-hidden="true">→</span>
                  </>
                )}
              </button>

              {error && (
                <div className="error-message" role="alert">
                  {error}
                </div>
              )}
            </form>

            <p className="privacy-note">
              <span aria-hidden="true">◈</span> Files are processed by your
              local backend and are not permanently saved by this prototype.
            </p>
          </div>

          <div className="panel answer-panel">
            <div className="panel-heading">
              <span className="step-number">02</span>
              <div>
                <h2>Answer & sources</h2>
                <p>Review the response and its supporting text.</p>
              </div>
            </div>

            {!result && !loading && (
              <div className="empty-state">
                <div className="empty-icon">✳</div>
                <h3>Your answer starts here</h3>
                <p>
                  Your document-grounded answer and relevant passages will
                  appear here after you ask a question.
                </p>
                <div className="empty-hint">
                  <span>01</span> Upload your document
                  <br />
                  <span>02</span> Ask a question
                  <br />
                  <span>03</span> Review the sources
                </div>
              </div>
            )}

            {loading && (
              <div className="loading-state" aria-live="polite">
                <span className="large-spinner" />
                <h3>Reading your document</h3>
                <p>
                  Retrieving relevant passages and generating an answer with
                  the local AI model. This may take a little time.
                </p>
              </div>
            )}

            {result && (
              <div className="result-content" aria-live="polite">
                <div className="answer-label">
                  <span className="answer-check">✓</span>
                  GENERATED ANSWER
                </div>

                <div className="answer-box">
                  <p>{result.answer}</p>
                </div>

                <div className="result-meta">
                  <span>Document</span>
                  <strong>{result.filename}</strong>
                </div>

                <div className="sources-heading">
                  <div>
                    <h3>Source passages</h3>
                    <p>Text retrieved from your document</p>
                  </div>
                  <span className="source-count">
                    {result.sources?.length || 0} found
                  </span>
                </div>

                {result.sources?.map((source, index) => (
                  <article className="source-card" key={`${index}-${source.text}`}>
                    <div className="source-card-top">
                      <span>Source {index + 1}</span>
                      <span className="similarity-score">
                        Similarity: {Number(source.similarity).toFixed(3)}
                      </span>
                    </div>
                    <p>{source.text}</p>
                  </article>
                ))}

                <p className="verification-note">
                  Always verify the generated answer against its source.
                  Similarity scores are not confidence percentages.
                </p>
              </div>
            )}
          </div>
        </section>

        <footer className="page-footer">
          <span>DocuMind AI</span>
          <span>Built with React, FastAPI and open-source models</span>
        </footer>
      </main>
    </div>
  );
}

export default App;