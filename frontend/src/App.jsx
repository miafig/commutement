import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Home from './pages/Home';
import Record from './pages/Record';
import Train from './pages/Train';
import Get from './pages/Get';
import "./App.css";

function App() {
  return (
    <Router>
      <nav>
        <Link to="/commutement">Home</Link>
        <Link to="/commutement/record">Record</Link>
        <Link to="/commutement/train">Train</Link>
        <Link to="/commutement/get">Get</Link>
      </nav>
      
      <Routes>
        <Route path="/commutement" element={<Home />} />
        <Route path="/commutement/record" element={<Record />} />
        <Route path="/commutement/train" element={<Train />} />
        <Route path="/commutement/get" element={<Get />} />
      </Routes>
    </Router>
  );
}
export default App;
