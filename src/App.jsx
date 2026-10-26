
import { useState } from "react";
import {
  Routes,
  Route,
  Link,
  useNavigate,
} from "react-router-dom";

import Home from "./pages/Home";
import Login from "./pages/Login";
import Cart from "./pages/Cart";
import Payment from "./pages/Payment";
import Success from "./pages/Success";

function App() {
  const [cart, setCart] = useState([]);
  const [user, setUser] = useState(null);

  const navigate = useNavigate();

  function addToCart(product) {
    setCart((previousCart) => {
      const existing = previousCart.find(
        (item) => item.id === product.id
      );

      if (existing) {
        return previousCart.map((item) =>
          item.id === product.id
            ? { ...item, quantity: item.quantity + 1 }
            : item
        );
      }

      return [...previousCart, { ...product, quantity: 1 }];
    });
  }

  function removeFromCart(productId) {
    setCart((previousCart) =>
      previousCart.filter((item) => item.id !== productId)
    );
  }

  function checkout() {
    if (cart.length === 0) {
      alert("Your cart is empty!");
      return;
    }

    navigate(user ? "/payment" : "/login");
  }

  function login(userDetails) {
    setUser(userDetails);
    navigate("/payment");
  }

  function completePayment() {
  console.log("Payment successful!");

  setCart([]);
  navigate("/success");
  }

  return (
    <div>
      <nav>
        <Link to="/">Home</Link>{" | "}
        <Link to="/cart">
          Cart ({cart.reduce(
            (total, item) => total + item.quantity, 0
          )})
        </Link>{" | "}
        <Link to="/login">Login</Link>{" | "}
        <Link to="/payment">Payment</Link>
      </nav>

      <hr />

      <Routes>
        <Route
          path="/"
          element={<Home onAddToCart={addToCart} />}
        />

        <Route
          path="/cart"
          element={
            <Cart
              cart={cart}
              onRemove={removeFromCart}
              onCheckout={checkout}
            />
          }
        />

        <Route
          path="/login"
          element={<Login onLogin={login} />}
        />

        <Route
          path="/payment"
          element={
            <Payment
              cart={cart}
              user={user}
              onPaymentSuccess={completePayment}
            />
          }
        />

        <Route path="/success" element={<Success />} />

        <Route
          path="*"
          element={<h1>404 - Page Not Found</h1>}
        />
      </Routes>
    </div>
  );
}

export default App;
