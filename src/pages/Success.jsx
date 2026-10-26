import { Link } from "react-router-dom";

function Success() {
  console.log("Success page rendered!");

  return (
    <div>
      <h1>Order Placed Successfully!</h1>
      <p>Thank you for shopping with us.</p>
      <Link to="/">Continue Shopping</Link>
    </div>
  );
}

export default Success;