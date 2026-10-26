
function Cart({ cart, onRemove, onCheckout }) {
  const totalPrice = cart.reduce(
    (total, item) => total + item.price * item.quantity,
    0
  );

  return (
    <div>
      <h1>Your Shopping Cart</h1>

      {cart.length === 0 ? (
        <p>Your cart is empty.</p>
      ) : (
        <>
          {cart.map((item) => (
            <div className="product-card" key={item.id}>
              <h3>{item.name}</h3>
              <p>Price: ₹{item.price}</p>
              <p>Quantity: {item.quantity}</p>
              <p>
                Subtotal: ₹{item.price * item.quantity}
              </p>

              <button onClick={() => onRemove(item.id)}>
                Remove
              </button>
            </div>
          ))}

          <h2>Total: ₹{totalPrice}</h2>

          <button onClick={onCheckout}>
            Proceed to Checkout
          </button>
        </>
      )}
    </div>
  );
}

export default Cart;
