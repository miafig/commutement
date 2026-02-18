import React, { useState } from "react";
import "./App.css";

export default function App() {
  const [form, setForm] = useState({
    dayOfWeek: "",
    departureTime: "08:00",
    arrivalTime: "09:00",
    transport: "train",
    route: "paddington",
    stops: [],
    notes: "",
    pleasureRating: 5,
  });

  const [status, setStatus] = useState("");
  const [count, setCount] = useState(0);

  // Fetch entry count on load
  React.useEffect(() => {
    fetchCount();
  }, []);

  const fetchCount = async () => {
    try {
      const res = await fetch("http://localhost:5000/api/commutes");
      const data = await res.json();
      setCount(data.count || 0);
    } catch (e) {
      console.error("Failed to fetch count", e);
    }
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;

    if (type === "checkbox") {
      const newStops = checked
        ? [...form.stops, value]
        : form.stops.filter((s) => s !== value);
      setForm({ ...form, stops: newStops });
    } else {
      setForm({ ...form, [name]: value });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!form.dayOfWeek) {
      setStatus("error: select a day");
      return;
    }

    try {
      const res = await fetch("http://localhost:5000/api/commute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });

      if (res.ok) {
        setStatus("✓ saved");
        setForm({
          dayOfWeek: "",
          departureTime: "08:00",
          arrivalTime: "09:00",
          transport: "train",
          route: "paddington",
          stops: [],
          notes: "",
          pleasureRating: 5,
        });
        fetchCount();
        setTimeout(() => setStatus(""), 3000);
      } else {
        setStatus("error: failed to save");
      }
    } catch (e) {
      setStatus("error: connection failed");
    }
  };

  return (
    <div className="container">
      <div className="card">
        <h1>Record Commute</h1>
        <p className="subtitle">{count} entries saved</p>

        {status && <div className={`status ${status.startsWith("error") ? "error" : "success"}`}>{status}</div>}

        <form onSubmit={handleSubmit}>
          <div className="field">
            <label>Day *</label>
            <select name="dayOfWeek" value={form.dayOfWeek} onChange={handleChange}>
              <option value="">Select...</option>
              <option value="Monday">Monday</option>
              <option value="Tuesday">Tuesday</option>
              <option value="Wednesday">Wednesday</option>
              <option value="Thursday">Thursday</option>
              <option value="Friday">Friday</option>
              <option value="Saturday">Saturday</option>
              <option value="Sunday">Sunday</option>
            </select>
          </div>

          <div className="row">
            <div className="field">
              <label>Departure</label>
              <input type="time" name="departureTime" value={form.departureTime} onChange={handleChange} />
            </div>
            <div className="field">
              <label>Arrival</label>
              <input type="time" name="arrivalTime" value={form.arrivalTime} onChange={handleChange} />
            </div>
          </div>

          <div className="field">
            <label>Transport</label>
            <div className="radio">
              <label>
                <input type="radio" name="transport" value="bike" checked={form.transport === "bike"} onChange={handleChange} />
                Bike
              </label>
              <label>
                <input type="radio" name="transport" value="train" checked={form.transport === "train"} onChange={handleChange} />
                Train
              </label>
            </div>
          </div>

          <div className="field">
            <label>Route</label>
            <select name="route" value={form.route} onChange={handleChange}>
              <option value="paddington">Paddington</option>
              <option value="monument">Monument</option>
              <option value="paddington+walk">Paddington + Walk</option>
              <option value="monument+walk">Monument + Walk</option>
              <option value="other">Other</option>
            </select>
          </div>

          <div className="field">
            <label>Stops</label>
            <div className="checkbox">
              {["coffee", "breakfast", "lunch", "walk", "none"].map((stop) => (
                <label key={stop}>
                  <input type="checkbox" value={stop} checked={form.stops.includes(stop)} onChange={handleChange} />
                  {stop}
                </label>
              ))}
            </div>
          </div>

          <div className="field">
            <label>
              Pleasure: <strong>{form.pleasureRating}</strong>
            </label>
            <input type="range" name="pleasureRating" min="1" max="10" value={form.pleasureRating} onChange={handleChange} />
          </div>

          <div className="field">
            <label>Notes</label>
            <textarea name="notes" value={form.notes} onChange={handleChange} placeholder="Optional observations..." rows="2" />
          </div>

          <button type="submit">Save</button>
        </form>
      </div>
    </div>
  );
}
