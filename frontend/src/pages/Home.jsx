import React, { useState } from "react";


export default function App() {
    return (
    <div className="container">
      <div className="card">
        <h1>this is commutement</h1>
        <p className="subtitle">commute enjoyment</p>
      </div>

      <div className="card">
        <h3>who</h3>
        <p>mia</p>
        <h3>what</h3>
        <p>a neural model for optimising daily commute for time and enjoyment</p>
        <h3>why</h3>
        <p>for fun</p>
        <h3>how</h3>
        <p>record my daily commutes for a few months and then train a neural network model to get the optimised commute</p>
      </div>

      <div className="card">
        <h3>use</h3>
        <p><strong>record</strong> a commute by filling in the details. this is saved in a database of commutes</p>
        <p><strong>train</strong> a commute by choosing the train parameters and (re)training the model</p>
        <p><strong>get</strong> an optimised commute: choose your parameters and run!</p>
      </div>

    </div>
    );
}
