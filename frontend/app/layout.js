import "./globals.css";

export const metadata = {
  title: "Edge AI Drug Classifier",
  description: "Vercel frontend for FastAPI PyTorch inference backend",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
