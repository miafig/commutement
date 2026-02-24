import React, { useState } from "react";

const apiUrl = "https://miafig-commutement.hf.space/api/";

const initState = {
    dayOfWeek: "",
    going: "",
    departureTime: "06:00",
    arrivalTime: "07:00",
    transport: "train",
    route: "",
    sideQuests: [],
    pleasureRating: 5,
    disruptions: "",
    company: "",
    rush: "",
}

export default function App() {
  const [form, setForm] = useState(initState);

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

    if (type === "checkbox") {
      if (name === "sideQuest") {
        const newSideQuests = checked
        ? [...form.sideQuests, value]
        : form.sideQuests.filter((s) => s !== value);
        setForm({ ...form, sideQuests: newSideQuests });
      }
      else if (name === "disruption") {
        const newDisruptions = checked
        ? [...form.disruptions, value]
        : form.disruptions.filter((d) => d !== value);
        setForm({ ...form, disruptions: newDisruptions });
      }
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
      const res = await fetch(apiUrl + "commute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });

      if (res.ok) {
        setStatus("✓ saved");
        setForm(initState);
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
        <h1>record a commute</h1>
        <p className="subtitle">{count} entries saved</p>

        {status && <div className={`status ${status.startsWith("error") ? "error" : "success"}`}>{status}</div>}

        <form onSubmit={handleSubmit}>
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
              pleasure <strong>{form.pleasureRating}</strong>
            </label>
            <input type="range" name="pleasureRating" min="1" max="10" value={form.pleasureRating} onChange={handleChange} />
          </div>

          <button type="submit">save</button>
        </form>
      </div>
    </div>
  );
}
