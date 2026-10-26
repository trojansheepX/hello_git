
import { useState } from "react";

function Login({ onLogin }) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");

  function handleSubmit(event) {
    event.preventDefault();

    if (!name.trim() || !email.trim()) {
      alert("Please enter your name and email.");
      return;
    }

    onLogin({
      name: name.trim(),
      email: email.trim(),
    });
  }

  return (
    <div>
      <h1>Login</h1>

      <form onSubmit={handleSubmit}>
        <div>
          <label>Name: </label>
          <input
            type="text"
            value={name}
            onChange={(event) => setName(event.target.value)}
            required
          />
        </div>

        <br />

        <div>
          <label>Email: </label>
          <input
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            required
          />
        </div>

        <br />

        <button type="submit">Continue to Payment</button>
      </form>
    </div>
  );
}

export default Login;
