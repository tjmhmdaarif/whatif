'use client';

import { Copy, CheckCircle, Clock, Zap, FileText, ExternalLink } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ResultsPanelProps {
  question: string;
  result: {
    answer: string;
    chunks: string[];
    latency_ms: number;
    llm_calls: number;
    citations?: Array<{ chunk_id: string; doc_id: string; text_snippet: string }>;
  };
  method: 'vector' | 'graph' | 'hybrid';
}

export function ResultsPanel({ question, result, method }: ResultsPanelProps) {
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  return (
    <div className="card animate-slide-up">
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className={cn(
              'p-2 rounded-lg',
              method === 'vector' && 'bg-blue-100 dark:bg-blue-900/30',
              method === 'graph' && 'bg-purple-100 dark:bg-purple-900/30',
              method === 'hybrid' && 'bg-gradient-to-r from-blue-500 to-purple-500'
            )}>
              <Zap className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-dark-900 dark:text-white">Answer</h2>
              <p className="text-sm text-dark-500 dark:text-dark-400">
                Retrieved {result.chunks.length} chunks in {result.latency_ms.toFixed(0)}ms
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => copyToClipboard(result.answer)}
              className="btn-ghost p-2"
              title="Copy answer"
            >
              <Copy className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Question */}
        <div className="bg-dark-50 dark:bg-dark-800/50 rounded-lg p-4">
          <p className="text-sm font-medium text-dark-500 dark:text-dark-400 mb-1">Question</p>
          <p className="text-dark-900 dark:text-white">{question}</p>
        </div>

        {/* Answer */}
        <div className="space-y-2">
          <p className="text-sm font-medium text-dark-500 dark:text-dark-400">Response</p>
          <div className="prose prose-dark max-w-none dark:prose-invert bg-white dark:bg-dark-800/50 rounded-lg p-4">
            <p className="whitespace-pre-wrap text-dark-900 dark:text-white">{result.answer || 'No answer generated'}</p>
          </div>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2 border-t border-dark-200 dark:border-dark-700">
          <div className="p-3 rounded-lg bg-dark-50 dark:bg-dark-800/50 text-center">
            <Clock className="w-5 h-5 text-primary-500 mx-auto mb-1" />
            <p className="text-2xl font-bold text-dark-900 dark:text-white">{result.latency_ms.toFixed(0)}ms</p>
            <p className="text-xs text-dark-500 dark:text-dark-400">Latency</p>
          </div>
          <div className="p-3 rounded-lg bg-dark-50 dark:bg-dark-800/50 text-center">
            <Zap className="w-5 h-5 text-primary-500 mx-auto mb-1" />
            <p className="text-2xl font-bold text-dark-900 dark:text-white">{result.llm_calls}</p>
            <p className="text-xs text-dark-500 dark:text-dark-400">LLM Calls</p>
          </div>
          <div className="p-3 rounded-lg bg-dark-50 dark:bg-dark-800/50 text-center">
            <FileText className="w-5 h-5 text-primary-500 mx-auto mb-1" />
            <p className="text-2xl font-bold text-dark-900 dark:text-white">{result.chunks.length}</p>
            <p className="text-xs text-dark-500 dark:text-dark-400">Chunks</p>
          </div>
          <div className="p-3 rounded-lg bg-dark-50 dark:bg-dark-800/50 text-center">
            <Zap className="w-5 h-5 text-primary-500 mx-auto mb-1" />
            <p className="text-2xl font-bold text-dark-900 dark:text-white">
              {method.charAt(0).toUpperCase() + method.slice(1)}
            </p>
            <p className="text-xs text-dark-500 dark:text-dark-400">Mode</p>
          </div>
        </div>

        {/* Citations / Retrieved Chunks */}
        {result.citations && result.citations.length > 0 && (
          <div className="pt-2 border-t border-dark-200 dark:border-dark-700">
            <p className="text-sm font-medium text-dark-500 dark:text-dark-400 mb-3 flex items-center gap-1">
              <FileText className="w-4 h-4" />
              Retrieved Chunks ({result.citations.length})
            </p>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {result.citations.map((citation, i) => (
                <div key={i} className="bg-dark-50 dark:bg-dark-800/50 rounded-lg p-3">
                  <div className="flex items-start justify-between gap-2 mb-1">
                    <span className="text-xs font-mono text-primary-600 dark:text-primary-400">
                      {citation.chunk_id}
                    </span>
                    <span className="text-xs text-dark-400 dark:text-dark-500">
                      Doc: {citation.doc_id}
                    </span>
                  </div>
                  <p className="text-sm text-dark-600 dark:text-dark-300 line-clamp-2">
                    {citation.text_snippet}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Fallback: Chunk IDs */}
        {(!result.citations || result.citations.length === 0) && result.chunks.length > 0 && (
          <div className="pt-2 border-t border-dark-200 dark:border-dark-700">
            <p className="text-sm font-medium text-dark-500 dark:text-dark-400 mb-3 flex items-center gap-1">
              <FileText className="w-4 h-4" />
              Chunk IDs ({result.chunks.length})
            </p>
            <div className="flex flex-wrap gap-2">
              {result.chunks.slice(0, 10).map((chunkId, i) => (
                <span key={i} className="px-2 py-1 text-xs rounded bg-dark-100 dark:bg-dark-800 text-dark-700 dark:text-dark-300 font-mono">
                  {chunkId}
                </span>
              ))}
              {result.chunks.length > 10 && (
                <span className="px-2 py-1 text-xs rounded bg-dark-100 dark:bg-dark-800 text-dark-500 dark:text-dark-400">
                  +{result.chunks.length - 10} more
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}