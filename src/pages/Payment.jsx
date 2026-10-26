
import { Navigate } from "react-router-dom";

function Payment({ cart, user, onPaymentSuccess }) {
  if (!user) {
    return <Navigate to="/login" replace />;
  }

  if (cart.length === 0) {
    return <Navigate to="/" replace />;
  }

  const totalPrice = cart.reduce(
    (total, item) => total + item.price * item.quantity,
    0
  );

  return (
    <div>
      <h1>Payment</h1>

      <p>Welcome, {user.name}!</p>
      <p>Email: {user.email}</p>

      <h2>Order Summary</h2>

      {cart.map((item) => (
        <p key={item.id}>
          {item.name} × {item.quantity} = ₹
          {item.price * item.quantity}
        </p>
      ))}

      <h3>Total Amount: ₹{totalPrice}</h3>

      <button onClick={onPaymentSuccess}>
        Pay ₹{totalPrice}
      </button>

      <p>
        Demo only: clicking Pay simulates a successful
        payment. No money is charged.
      </p>
    </div>
  );
}

export default Payment;
