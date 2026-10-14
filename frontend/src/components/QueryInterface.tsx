'use client';

import { useState, FormEvent } from 'react';
import { Search, Loader2, Sparkles, ArrowRight, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface QueryInterfaceProps {
  method: 'vector' | 'graph' | 'hybrid';
  onQuery: (question: string) => void;
  disabled?: boolean;
}

const EXAMPLES = [
  'Who founded Microsoft?',
  'Where is the Eiffel Tower located?',
  'When was Barack Obama born?',
  'What company did Steve Jobs co-found?',
  'Which country is Tokyo the capital of?',
];

export function QueryInterface({ method, onQuery, disabled }: QueryInterfaceProps) {
  const [question, setQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!question.trim() || isLoading || disabled) return;
    setIsLoading(true);
    onQuery(question);
    setIsLoading(false);
  };

  return (
    <form onSubmit={handleSubmit} className="card p-6 animate-slide-up">
      <div className="flex flex-col gap-4">
        {/* Method Badge */}
        <div className="flex items-center gap-3">
          <div className={cn(
            'px-3 py-1 rounded-full text-xs font-medium',
            method === 'vector' && 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400',
            method === 'graph' && 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400',
            method === 'hybrid' && 'bg-gradient-to-r from-blue-500 to-purple-500 text-white'
          )}>
            {method.charAt(0).toUpperCase() + method.slice(1)} Mode
          </div>
          <span className="text-sm text-dark-500 dark:text-dark-400">
            {method === 'vector' && 'Semantic search only'}
            {method === 'graph' && 'Knowledge graph traversal'}
            {method === 'hybrid' && 'Vector + Graph with RRF fusion'}
          </span>
        </div>

        {/* Question Input */}
        <div className="relative">
          <label htmlFor="question" className="sr-only">Your question</label>
          <textarea
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a multi-hop question... (e.g., 'Who founded Microsoft and where are they from?')"
            rows={3}
            className={cn(
              'input resize-none pr-12',
              disabled && 'opacity-50 cursor-not-allowed'
            )}
            disabled={disabled || isLoading}
            aria-label="Enter your question"
          />
          <button
            type="submit"
            onClick={handleSubmit}
            disabled={!question.trim() || isLoading || disabled}
            className="absolute right-3 bottom-3 btn-primary disabled:opacity-50"
            aria-label="Submit query"
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <>
                <Search className="w-5 h-5 mr-2" />
                Query
              </>
            )}
          </button>
        </div>

        {/* Example Questions */}
        {!disabled && (
          <div className="pt-2 border-t border-dark-200 dark:border-dark-700">
            <p className="text-xs text-dark-500 dark:text-dark-400 mb-2 flex items-center gap-1">
              <Sparkles className="w-3 h-3" />
              Example questions:
            </p>
            <div className="flex flex-wrap gap-2">
              {EXAMPLES.map((ex, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => setQuestion(ex)}
                  className="px-3 py-1.5 text-xs rounded-lg bg-dark-100 dark:bg-dark-800 hover:bg-dark-200 dark:hover:bg-dark-700 text-dark-700 dark:text-dark-300 transition-colors"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        {disabled && (
          <div className="flex items-center gap-2 text-sm text-yellow-600 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded-lg">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span>Build an index first using the CLI, then refresh the page.</span>
          </div>
        )}
      </div>
    </form>
  );
}