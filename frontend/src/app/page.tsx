'use client';

import { useState, useCallback } from 'react';
import { Search, Loader2, CheckCircle, AlertCircle, Info, Sparkles, Brain, Database, Zap } from 'lucide-react';
import { QueryInterface } from '@/components/QueryInterface';
import { ResultsPanel } from '@/components/ResultsPanel';
import { Sidebar } from '@/components/Sidebar';

interface QueryResult {
  answer: string;
  chunks: string[];
  latency_ms: number;
  llm_calls: number;
  citations?: Array<{ chunk_id: string; doc_id: string; text_snippet: string }>;
}

interface HealthStatus {
  status: string;
  mode: string;
  index_loaded: boolean;
}

export default function Home() {
  const [method, setMethod] = useState<'vector' | 'graph' | 'hybrid'>('hybrid');
  const [topK, setTopK] = useState(10);
  const [maxHops, setMaxHops] = useState(2);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [lastQuery, setLastQuery] = useState<{ question: string; result: QueryResult } | null>(null);

  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch('/api/backend/health');
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
      }
    } catch {
      setHealth({ status: 'error', mode: 'unknown', index_loaded: false });
    }
  }, []);

  const handleQuery = async (question: string) => {
    try {
      const res = await fetch('/api/backend/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, method, top_k: topK, max_hops: maxHops }),
      });
      if (res.ok) {
        const data = await res.json();
        setLastQuery({ question, result: data });
      }
    } catch (error) {
      console.error('Query failed:', error);
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-dark-200 bg-white/80 backdrop-blur-sm sticky top-0 z-40 dark:bg-dark-900/80 dark:border-dark-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary-600 rounded-lg">
                <Brain className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-dark-900 dark:text-white">GraphRAG-MultiHop</h1>
                <p className="text-xs text-dark-500 dark:text-dark-400">Graph-enhanced RAG for Multi-hop QA</p>
              </div>
            </div>
            
            {/* Health Status */}
            <div className="flex items-center gap-4">
              <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium ${
                health?.index_loaded 
                  ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' 
                  : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400'
              }`}>
                {health?.index_loaded ? (
                  <>
                    <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span>
                    Index Ready
                  </>
                ) : (
                  <>
                    <span className="w-1.5 h-1.5 rounded-full bg-yellow-500 animate-pulse"></span>
                    No Index
                  </>
                )}
              </div>
              <button 
                onClick={checkHealth}
                className="p-2 rounded-lg hover:bg-dark-100 dark:hover:bg-dark-800 transition-colors"
                title="Refresh status"
              >
                <Zap className="w-4 h-4 text-dark-500" />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar */}
          <aside className="lg:col-span-1">
            <Sidebar
              method={method}
              setMethod={setMethod}
              topK={topK}
              setTopK={setTopK}
              maxHops={maxHops}
              setMaxHops={setMaxHops}
              health={health}
              onRebuild={checkHealth}
            />
          </aside>

          {/* Main Area */}
          <div className="lg:col-span-3 space-y-6">
            {/* Query Interface */}
            <QueryInterface
              method={method}
              onQuery={handleQuery}
              disabled={!health?.index_loaded}
            />

            {/* Results Panel */}
            {lastQuery && (
              <ResultsPanel
                question={lastQuery.question}
                result={lastQuery.result}
                method={method}
              />
            )}

            {/* Empty State */}
            {!lastQuery && health?.index_loaded && (
              <div className="card p-12 text-center animate-fade-in">
                <Sparkles className="w-16 h-16 mx-auto text-primary-500 mb-4" />
                <h2 className="text-2xl font-bold text-dark-900 dark:text-white mb-2">
                  Ready to Answer Questions
                </h2>
                <p className="text-dark-600 dark:text-dark-400 mb-6 max-w-md mx-auto">
                  Ask a multi-hop question and GraphRAG will combine vector search with 
                  knowledge graph traversal to find the answer.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 max-w-2xl mx-auto">
                  {[
                    { icon: Database, title: 'Vector Search', desc: 'Semantic similarity' },
                    { icon: Brain, title: 'Graph Traversal', desc: 'Multi-hop reasoning' },
                    { icon: Zap, title: 'Hybrid Fusion', desc: 'Best of both worlds' },
                  ].map((item, i) => (
                    <div key={i} className="p-4 rounded-lg bg-dark-50 dark:bg-dark-800/50 text-left">
                      <item.icon className="w-6 h-6 text-primary-500 mb-2" />
                      <h3 className="font-semibold text-dark-900 dark:text-white">{item.title}</h3>
                      <p className="text-sm text-dark-500 dark:text-dark-400">{item.desc}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {!health?.index_loaded && (
              <div className="card p-12 text-center animate-fade-in border-yellow-200 dark:border-yellow-800">
                <AlertCircle className="w-16 h-16 mx-auto text-yellow-500 mb-4" />
                <h2 className="text-2xl font-bold text-dark-900 dark:text-white mb-2">
                  No Index Loaded
                </h2>
                <p className="text-dark-600 dark:text-dark-400 mb-6 max-w-md mx-auto">
                  Build an index first using the CLI, then refresh this page.
                </p>
                <div className="flex flex-col sm:flex-row gap-3 justify-center">
                  <code className="px-4 py-2 bg-dark-100 dark:bg-dark-800 rounded-lg text-sm font-mono text-dark-700 dark:text-dark-300">
                    python -m src.graphrag.cli index --limit 50
                  </code>
                  <button
                    onClick={checkHealth}
                    className="btn-secondary"
                  >
                    Refresh Status
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-dark-200 bg-white/50 dark:bg-dark-900/50 dark:border-dark-700 py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-sm text-dark-500 dark:text-dark-400">
          GraphRAG-MultiHop v0.1.0 • Built with Next.js, FastAPI, FAISS, NetworkX & sentence-transformers
        </div>
      </footer>
    </div>
  );
}