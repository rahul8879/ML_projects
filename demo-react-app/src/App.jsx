import { useMemo, useState } from "react";

const defaultPayload = {
  category: "food_dining",
  amt: 150.75,
  gender: "male",
  state: "CA",
  city_pop: 100000,
  job: "engineer",
  distance: 5.0,
  trans_hour: 12,
  trans_minute: 30,
  trans_second: 45,
};

function App() {
  const [mode, setMode] = useState("single");
  const [payload, setPayload] = useState(defaultPayload);
  const [bulkText, setBulkText] = useState("");
  const [result, setResult] = useState(null);
  const [bulkResults, setBulkResults] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const payloadKeys = useMemo(() => Object.keys(defaultPayload), []);
  const bulkExample = useMemo(
    () =>
      JSON.stringify(
        [
          defaultPayload,
          {
            ...defaultPayload,
            amt: 920.5,
            distance: 22.1,
            trans_hour: 2,
            trans_minute: 5,
            trans_second: 18,
          },
        ],
        null,
        2
      ),
    []
  );

  const updateField = (key, value) => {
    setPayload((prev) => ({ ...prev, [key]: value }));
  };

  const fillExample = () => {
    setPayload(defaultPayload);
    setBulkText(bulkExample);
    setResult(null);
    setBulkResults(null);
    setError(null);
  };

  const submitPrediction = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setBulkResults(null);
    try {
      const baseUrl = "http://127.0.0.1:8000";
      const endpoint = mode === "single" ? "/predict" : "/predict-bulk";
      const body =
        mode === "single"
          ? payload
          : (() => {
              const parsed = JSON.parse(bulkText);
              if (!Array.isArray(parsed) || parsed.length === 0) {
                throw new Error("Bulk input must be a non-empty JSON array.");
              }
              return parsed;
            })();

      const response = await fetch(`${baseUrl}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Request failed with ${response.status}`);
      }
      const data = await response.json();
      if (mode === "single") {
        setResult(data);
      } else {
        setBulkResults(data);
      }
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const switchMode = (nextMode) => {
    setMode(nextMode);
    setResult(null);
    setBulkResults(null);
    setError(null);
    if (nextMode === "bulk" && !bulkText) {
      setBulkText(bulkExample);
    }
    if (nextMode === "single") {
      setPayload(defaultPayload);
    }
  };

  return (
    <div className="page">
      <div className="hero">
        <span className="pill">Live Fraud Scoring</span>
        <h1>Credit Fraud Prediction</h1>
        <p>
          Fraud prediction helps flag suspicious transactions by learning patterns
          across amount, location, timing, and customer behavior. Use this demo to
          score transactions instantly, then explore more fraud analytics and
          production-ready AI solutions at rbyteai.com.
        </p>
        <div className="card">
          {result ? (
            <div className="result">
              <span className="badge">Prediction</span>
              <div className="value">
                {(result.fraud_probability * 100).toFixed(2)}%
              </div>
              <p>Class: {result.prediction === 1 ? "Fraud" : "Not Fraud"}</p>
            </div>
          ) : null}
          {bulkResults ? (
            <div className="result">
              <span className="badge">Bulk Predictions</span>
              <div className="result-list">
                {bulkResults.map((item, index) => (
                  <div className="result-row" key={index}>
                    <span>#{index + 1}</span>
                    <span>{(item.fraud_probability * 100).toFixed(2)}%</span>
                    <span>{item.prediction === 1 ? "Fraud" : "Not Fraud"}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
          {error ? (
            <div className="result">
              <span className="badge">Error</span>
              <p>{error}</p>
            </div>
          ) : null}
        </div>
      </div>
      <div className="card">
        <div className="card-head">
          <h2>Transaction Inputs</h2>
          <div className="mode-toggle">
            <button
              type="button"
              className={mode === "single" ? "toggle active" : "toggle"}
              onClick={() => switchMode("single")}
            >
              Single
            </button>
            <button
              type="button"
              className={mode === "bulk" ? "toggle active" : "toggle"}
              onClick={() => switchMode("bulk")}
            >
              Bulk
            </button>
          </div>
        </div>
        {mode === "single" ? (
          <div className="form-grid">
            {payloadKeys.map((key) => (
              <div key={key}>
                <label>{key.replace("_", " ")}</label>
                <input
                  value={payload[key]}
                  onChange={(e) => {
                    const rawValue = e.target.value;
                    const numericKeys = [
                      "amt",
                      "city_pop",
                      "distance",
                      "trans_hour",
                      "trans_minute",
                      "trans_second",
                    ];
                    const nextValue = numericKeys.includes(key)
                      ? Number(rawValue)
                      : rawValue;
                    updateField(key, nextValue);
                  }}
                />
              </div>
            ))}
          </div>
        ) : (
          <div className="bulk-input">
            <label>Bulk payload (JSON array)</label>
            <textarea
              rows={12}
              value={bulkText}
              onChange={(e) => setBulkText(e.target.value)}
            />
            <p className="hint">
              Send an array of transactions to <code>/predict-bulk</code>.
            </p>
          </div>
        )}
        <div className="actions">
          <button
            className="primary"
            onClick={submitPrediction}
            disabled={loading}
          >
            {loading
              ? "Scoring..."
              : mode === "single"
              ? "Predict Fraud"
              : "Predict Bulk"}
          </button>
          <button className="ghost" onClick={fillExample}>
            Use Example Data
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
