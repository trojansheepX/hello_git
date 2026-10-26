
import { useState } from "react";
import ProductCard from "../components/ProductCard";

const products = [
  { id: 1, name: "Laptop", price: 50000 },
  { id: 2, name: "Headphones", price: 2000 },
  { id: 3, name: "Keyboard", price: 1500 },
  { id: 4, name: "Mouse", price: 800 },
  { id: 5, name: "Monitor", price: 12000 },
];

function Home({ onAddToCart }) {
  const [search, setSearch] = useState("");

  const filteredProducts = products.filter((product) =>
    product.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div>
      <h1>Amazon</h1>

      <input
        type="text"
        placeholder="Search products..."
        value={search}
        onChange={(event) => setSearch(event.target.value)}
      />

      <div className="product-list">
        {filteredProducts.map((product) => (
          <ProductCard
            key={product.id}
            product={product}
            onAddToCart={onAddToCart}
          />
        ))}
      </div>

      {filteredProducts.length === 0 && (
        <p>No products found.</p>
      )}
    </div>
  );
}

export default Home;
