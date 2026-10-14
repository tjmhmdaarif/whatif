import type { Metadata, Viewport } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });

export const metadata: Metadata = {
  title: 'GraphRAG-MultiHop',
  description: 'Graph-enhanced Retrieval-Augmented Generation for Multi-hop QA',
  keywords: ['GraphRAG', 'RAG', 'Knowledge Graph', 'Multi-hop QA', 'AI', 'LLM'],
};

export const viewport: Viewport = {
  themeColor: '#0ea5e9',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} antialiased`}>
      <body className="min-h-screen flex flex-col">{children}</body>
    </html>
  );
}