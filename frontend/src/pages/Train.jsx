import React, { useState } from "react";

const apiUrl = "https://miafig-commutement.hf.space/api/";

const initTrainState = {
  epochs: 10,
  batch_size: 32,
  learning_rate: 0.001,
  train_val_split: 0.8,
}


export default function App() {
  const [form, setForm] = useState(initTrainState);
  const [status, setStatus] = useState("");
  const [count, setCount] = useState(0);

  // Fetch entry count on load
  React.useEffect(() => {
    fetchCount();
  }, []);

  const fetchCount = async () => {
    try {
      const res = await fetch(apiUrl + "commutes");
      const data = await res.json();
      setCount(data.count || 0);
    } catch (e) {
      console.error("Failed to fetch count", e);
    }
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setForm({ ...form, [name]: value });
  };

  const handleTrain = async (e) => {
    e.preventDefault();
    console.log(e)
    console.log(form)

    try {
      const res = await fetch(apiUrl + "train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      console.log(res)

      if (res.ok) {
        setStatus("✓ trained");
        setForm(initTrainState);
        setTimeout(() => setStatus(""), 6000);
      } else {
        setStatus("error: failed to train");
      }
    } catch (e) {
      setStatus("error: connection failed");
    }
  };

  return (
    <div className="container">
      <div className="card">
        <h1>train a commute</h1>
        <p className="subtitle">{count} data points</p>
    
      {status && <div className={`status ${status.startsWith("error") ? "error" : "success"}`}>{status}</div>}

      <form onSubmit={handleTrain}>
        <div className="field">
          <label>epochs <strong>{form.epochs}</strong></label>
          <input type="range" name="epochs" min="1" max="50" value={form.epochs} onChange={handleChange} />
        </div>

        <div className="field">
          <label>batch_size <strong>{form.batch_size}</strong></label>
          <input type="range" name="batch_size" min="1" max="50" value={form.batch_size} onChange={handleChange} />
        </div>

        <div className="field">
          <label>learning_rate <strong>{form.learning_rate}</strong></label>
          <input type="range" name="learning_rate" min="0.001" max="1" step="0.001" value={form.learning_rate} onChange={handleChange} />
        </div>

        <div className="field">
          <label>train_val_split <strong>{form.train_val_split}</strong></label>
          <input type="range" name="train_val_split" min="0" max="1" step="0.05" value={form.train_val_split} onChange={handleChange} />
        </div>

        <button type="submit">train</button>
      </form>
    </div>
    </div>
    );
}
