import React, { useState } from "react";

const apiUrl = "http://127.0.0.1:5000/api/" // "https://miafig-commutement.hf.space/api/";

const initPredState = {
  dayOfWeek: "",
  going: "",
  departureTime: "00:00",
  arrivalTime: "00:00",
  transport: "",
  route: "",
  sideQuests: [],
  disruptions: [],
  company: "",
  rush: "",
  samples: 50,
  timeWeight: 0.5,
  pleasureWeight: 0.5
}

export default function App() {
  const [form, setForm] = useState(initPredState);
  const [status, setStatus] = useState("");
  const [result, setResult] = useState(null);
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

    if (type === "checkbox") {
      if (name === "sideQuest") {
        const nextSideQuests = checked
          ? [...(form.sideQuests || []), value]
          : (form.sideQuests || []).filter((item) => item !== value);
        setForm({ ...form, sideQuests: nextSideQuests });
        return;
      }

      if (name === "disruption") {
        const nextDisruptions = checked
          ? [...(form.disruptions || []), value]
          : (form.disruptions || []).filter((item) => item !== value);
        setForm({ ...form, disruptions: nextDisruptions });
        return;
      }
    }

    setForm({ ...form, [name]: value });
  };

  const handlePred = async (e) => {
    e.preventDefault();
    setStatus("");
    setResult(null);

    const payload = {
      day: form.dayOfWeek,
      startTime: form.departureTime,
      known_features: {
        route: form.route,
      },
      inference_params: {
        n_samples: Number(form.samples),
        time_weight: Number(form.timeWeight),
        pleasure_weight: Number(form.pleasureWeight),
      },
    };

    try {
      const res = await fetch(apiUrl + "predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (res.ok) {
        setStatus("✓ predicted");
        setResult(data);
        setTimeout(() => setStatus(""), 6000);
      } else {
        setStatus(`error: ${data.error || "failed to predict"}`);
      }
    } catch (e) {
      setStatus("error: connection failed");
    }
  };

  return (
    <div className="container">
      <div className="card">
        <h1>get a commute</h1>
        <p className="subtitle">{count} data points</p>
    
      {status && <div className={`status ${status.startsWith("error") ? "error" : "success"}`}>{status}</div>}

      <form onSubmit={handlePred}>
          <div className="field">
            <label>day *</label>
            <select name="dayOfWeek" value={form.dayOfWeek} onChange={handleChange}>
              <option value="">...</option>
              <option value="Monday">monday</option>
              <option value="Tuesday">tuesday</option>
              <option value="Wednesday">wednesday</option>
              <option value="Thursday">thursday</option>
              <option value="Friday">friday</option>
            </select>
          </div>

          <div className="field">
            <label>going *</label>
            <div className="radio">
              <label>
                <input type="radio" name="going" value="home" checked={form.going === "home"} onChange={handleChange} />
                home
              </label>
              <label>
                <input type="radio" name="going" value="work" checked={form.going === "work"} onChange={handleChange} />
                work
              </label>
            </div>
          </div>

          <div className="row">
            <div className="field">
              <label>departure</label>
              <input type="time" name="departureTime" value={form.departureTime} onChange={handleChange} />
            </div>
            <div className="field">
              <label>arrival</label>
              <input type="time" name="arrivalTime" value={form.arrivalTime} onChange={handleChange} />
            </div>
          </div>

          <div className="field">
            <label>transport</label>
            <div className="radio">
              <label>
                <input type="radio" name="transport" value="bike" checked={form.transport === "bike"} onChange={handleChange} />
                bike
              </label>
              <label>
                <input type="radio" name="transport" value="train" checked={form.transport === "train"} onChange={handleChange} />
                train
              </label>
            </div>
          </div>

          <div className="field">
            <label>route</label>
            <select name="route" value={form.route} onChange={handleChange}>
              <option value="">empty</option>
              <option value="paddington">paddington</option>
              <option value="monument">monument</option>
              <option value="monument+walk">monument + walk</option>
              <option value="monument+circle">monument + circle</option>
              <option value="circle+hsk+walk">circle + hsk + walk</option>
              <option value="picadilly+central">picadilly + central</option>
              <option value="other">other</option>
            </select>
          </div>

          <div className="field">
            <label>side quests</label>
            <div className="checkbox">
              {["drink", "breakfast", "lunch", "walk", "grocery", "shop", "errand"].map((sideQuest) => (
                <label key={sideQuest}>
                  <input type="checkbox" name="sideQuest" value={sideQuest} checked={form.sideQuests.includes(sideQuest)} onChange={handleChange} />
                  {sideQuest}
                </label>
              ))}
            </div>
          </div>

          <div className="field">
            <label>disruptions</label>
            <div className="checkbox">
              {["strikes", "delay", "holiday", "late"].map((disruption) => (
                <label key={disruption}>
                  <input type="checkbox" name="disruption" value={disruption} checked={form.disruptions.includes(disruption)} onChange={handleChange} />
                  {disruption}
                </label>
              ))}
            </div>
          </div>

          <div className="field">
            <label>company</label>
            <div className="radio">
              <label>
                <input type="radio" name="company" value="yes" checked={form.company === "yes"} onChange={handleChange} />
                yes
              </label>
              <label>
                <input type="radio" name="company" value="no" checked={form.company === "no"} onChange={handleChange} />
                no
              </label>
            </div>
          </div>

          <div className="field">
            <label>rush</label>
            <div className="radio">
              <label>
                <input type="radio" name="rush" value="low" checked={form.rush === "low"} onChange={handleChange} />
                low
              </label>
              <label>
                <input type="radio" name="rush" value="medium" checked={form.rush === "medium"} onChange={handleChange} />
                medium
              </label>
              <label>
                <input type="radio" name="rush" value="high" checked={form.rush === "high"} onChange={handleChange} />
                high
              </label>
            </div>
          </div>

          <div className="field">
            <label>
              time weight <strong>{form.timeWeight}</strong>
            </label>
            <input type="range" name="timeWeight" min="0" max="1" step="0.1" value={form.timeWeight} onChange={handleChange} />
          </div>

          <div className="field">
            <label>
              pleasure weight <strong>{form.pleasureWeight}</strong>
            </label>
            <input type="range" name="pleasureWeight" min="0" max="1" step="0.1" value={form.pleasureWeight} onChange={handleChange} />
          </div>

          <button type="submit">predict</button>
          {result && (
            <div className="prediction-result">
              <h2>prediction result</h2>
              <p><strong>score:</strong> {result.best_score?.toFixed(3)}</p>
              <p><strong>travel time:</strong> {result.predicted_travel_time}</p>
              <p><strong>pleasure:</strong> {result.predicted_pleasure}</p>
              <div className="recommendation">
                <p><strong>day:</strong> {result.recommendation?.dayOfWeek}</p>
                <p><strong>departure:</strong> {result.recommendation?.departureTime}</p>
                <p><strong>arrival:</strong> {result.recommendation?.arrivalTime}</p>
                <p><strong>transport:</strong> {result.recommendation?.transport}</p>
                <p><strong>route:</strong> {result.recommendation?.route}</p>
                <p><strong>side quests:</strong> {(result.recommendation?.sideQuests || []).join(", ") || "none"}</p>
                <p><strong>disruptions:</strong> {(result.recommendation?.disruptions || []).join(", ") || "none"}</p>
                <p><strong>company:</strong> {result.recommendation?.company}</p>
                <p><strong>rush:</strong> {result.recommendation?.rush}</p>
              </div>
            </div>
          )}
        </form>
    </div>
    </div>
    );
}
