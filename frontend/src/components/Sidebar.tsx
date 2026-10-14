'use client';

import { Database, Brain, Zap, Settings, RefreshCw, AlertCircle, CheckCircle, Info } from 'lucide-react';
import { cn } from '@/lib/utils';

interface SidebarProps {
  method: 'vector' | 'graph' | 'hybrid';
  setMethod: (method: 'vector' | 'graph' | 'hybrid') => void;
  topK: number;
  setTopK: (topK: number) => void;
  maxHops: number;
  setMaxHops: (maxHops: number) => void;
  health: { status: string; mode: string; index_loaded: boolean } | null;
  onRebuild: () => void;
}

const METHOD_OPTIONS = [
  { value: 'vector' as const, label: 'Vector', icon: Database, desc: 'Semantic search only', color: 'blue' },
  { value: 'graph' as const, label: 'Graph', icon: Brain, desc: 'Knowledge graph traversal', color: 'purple' },
  { value: 'hybrid' as const, label: 'Hybrid', icon: Zap, desc: 'Vector + Graph with RRF', color: 'gradient' },
];

export function Sidebar({ 
  method, 
  setMethod, 
  topK, 
  setTopK, 
  maxHops, 
  setMaxHops, 
  health, 
  onRebuild 
}: SidebarProps) {
  return (
    <div className="space-y-6">
      {/* Retrieval Method */}
      <div className="card p-4">
        <h3 className="font-semibold text-dark-900 dark:text-white mb-4 flex items-center gap-2">
          <Settings className="w-5 h-5 text-primary-500" />
          Retrieval Method
        </h3>
        <div className="space-y-2">
          {METHOD_OPTIONS.map(({ value, label, icon: Icon, desc, color }) => (
            <button
              key={value}
              onClick={() => setMethod(value)}
              className={cn(
                'w-full p-3 rounded-lg border-2 transition-all duration-200 text-left flex items-center gap-3',
                method === value
                  ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                  : 'border-dark-200 dark:border-dark-700 hover:border-primary-300 dark:hover:border-primary-700'
              )}
            >
              <div className={cn(
                'p-2 rounded-lg',
                color === 'blue' && 'bg-blue-100 dark:bg-blue-900/30',
                color === 'purple' && 'bg-purple-100 dark:bg-purple-900/30',
                color === 'gradient' && 'bg-gradient-to-r from-blue-500 to-purple-500'
              )}>
                <Icon className="w-5 h-5 text-white" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="font-medium text-dark-900 dark:text-white">{label}</p>
                <p className="text-xs text-dark-500 dark:text-dark-400 truncate">{desc}</p>
              </div>
              {method === value && (
                <CheckCircle className="w-5 h-5 text-primary-500" />
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Parameters */}
      <div className="card p-4">
        <h3 className="font-semibold text-dark-900 dark:text-white mb-4 flex items-center gap-2">
          <Settings className="w-5 h-5 text-primary-500" />
          Parameters
        </h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-dark-700 dark:text-dark-300 mb-1">
              Top-K Chunks: {topK}
            </label>
            <input
              type="range"
              min="1"
              max="20"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-full h-2 bg-dark-200 dark:bg-dark-700 rounded-lg appearance-none cursor-pointer accent-primary-500"
            />
            <p className="text-xs text-dark-500 dark:text-dark-400 mt-1">
              Number of chunks to retrieve
            </p>
          </div>
          <div>
            <label className="block text-sm font-medium text-dark-700 dark:text-dark-300 mb-1">
              Max Graph Hops: {maxHops}
            </label>
            <input
              type="range"
              min="1"
              max="3"
              value={maxHops}
              onChange={(e) => setMaxHops(Number(e.target.value))}
              className="w-full h-2 bg-dark-200 dark:bg-dark-700 rounded-lg appearance-none cursor-pointer accent-primary-500"
            />
            <p className="text-xs text-dark-500 dark:text-dark-400 mt-1">
              Graph traversal depth (graph/hybrid only)
            </p>
          </div>
        </div>
      </div>

      {/* Index Status */}
      <div className="card p-4">
        <h3 className="font-semibold text-dark-900 dark:text-white mb-4 flex items-center gap-2">
          <Database className="w-5 h-5 text-primary-500" />
          Index Status
        </h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-600 dark:text-dark-400">Vector Index</span>
            <span className={cn(
              'px-2 py-0.5 rounded-full text-xs font-medium',
              health?.index_loaded
                ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400'
                : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400'
            )}>
              {health?.index_loaded ? 'Loaded' : 'Missing'}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-600 dark:text-dark-400">Graph Store</span>
            <span className={cn(
              'px-2 py-0.5 rounded-full text-xs font-medium',
              health?.index_loaded
                ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400'
                : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400'
            )}>
              {health?.index_loaded ? 'Ready' : 'Not Built'}
            </span>
          </div>
          <button
            onClick={onRebuild}
            disabled={!health?.index_loaded}
            className="w-full btn-secondary"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Rebuild Index (CLI)
          </button>
        </div>
      </div>

      {/* Info */}
      <div className="card p-4">
        <h3 className="font-semibold text-dark-900 dark:text-white mb-3 flex items-center gap-2">
          <Info className="w-5 h-5 text-primary-500" />
          How it Works
        </h3>
        <ul className="space-y-2 text-sm text-dark-600 dark:text-dark-400">
          <li className="flex items-start gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-primary-500 mt-1.5 flex-shrink-0" />
            <strong>Vector:</strong> Semantic similarity search using FAISS
          </li>
          <li className="flex items-start gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-purple-500 mt-1.5 flex-shrink-0" />
            <strong>Graph:</strong> Multi-hop reasoning via knowledge graph
          </li>
          <li className="flex items-start gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 mt-1.5 flex-shrink-0" />
            <strong>Hybrid:</strong> RRF fusion of both approaches
          </li>
        </ul>
      </div>

      {/* System Info */}
      <div className="card p-4">
        <h3 className="font-semibold text-dark-900 dark:text-white mb-3 flex items-center gap-2">
          <Info className="w-5 h-5 text-primary-500" />
          System
        </h3>
        <dl className="space-y-2 text-sm">
          <div className="flex justify-between">
            <dt className="text-dark-500 dark:text-dark-400">Embedding</dt>
            <dd className="font-mono text-dark-900 dark:text-white">BAAI/bge-small-en-v1.5</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-dark-500 dark:text-dark-400">Vector Store</dt>
            <dd className="font-mono text-dark-900 dark:text-white">FAISS</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-dark-500 dark:text-dark-400">Graph Store</dt>
            <dd className="font-mono text-dark-900 dark:text-white">NetworkX</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-dark-500 dark:text-dark-400">LLM</dt>
            <dd className="font-mono text-dark-900 dark:text-white">Mock (for testing)</dd>
          </div>
        </dl>
      </div>
    </div>
  );
}