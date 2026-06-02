import { Newspaper } from 'lucide-react';
import { useMemo, useState } from 'react';
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

import { classify } from '../api/api.js';
import ErrorMessage from './ErrorMessage.jsx';
import LoadingButton from './LoadingButton.jsx';
import PresetButtons from './PresetButtons.jsx';

const examples = [
  'Global leaders meet to discuss a new climate agreement',
  'The national team wins the championship after a dramatic final',
  'Technology stocks rise as chip demand increases',
  'A new spacecraft sends detailed images from Mars',
];

export default function ClassificationTab() {
  const [input, setInput] = useState(examples[2]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const chartData = useMemo(() => {
    if (!result?.scores) return [];
    return Object.entries(result.scores)
      .map(([name, score]) => ({
        name,
        score: Number((score * 100).toFixed(2)),
      }))
      .sort((left, right) => right.score - left.score);
  }, [result]);

  const submit = async () => {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const response = await classify(input);
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Classification request failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
      <div className="space-y-4">
        <div>
          <h2 className="text-xl font-semibold">AG News classification</h2>
          <p className="mt-1 text-sm text-[#607064]">Four-class news classifier with full probability scores.</p>
        </div>

        <PresetButtons examples={examples} onSelect={setInput} />

        <label className="block">
          <span className="text-sm font-semibold">News text</span>
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            rows={7}
            className="mt-2 w-full resize-y rounded-md border border-[#cdd8cc] px-3 py-3 text-sm outline-none transition focus:border-[#1f5f46] focus:ring-2 focus:ring-[#d7eadf]"
          />
        </label>

        <LoadingButton onClick={submit} loading={loading} disabled={!input.trim()}>
          <Newspaper className="h-4 w-4" aria-hidden="true" />
          Classify
        </LoadingButton>
        <ErrorMessage message={error} />
      </div>

      <div className="min-h-[22rem] rounded-md border border-[#d7ded4] bg-[#f8faf7] p-4">
        <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-[#607064]">Output</h3>
        {!result && <p className="mt-4 text-sm text-[#607064]">Class scores will appear here.</p>}
        {result && (
          <div className="mt-4 space-y-5">
            <div className="inline-flex items-center gap-2 rounded-md bg-[#e6f1ea] px-3 py-2 text-sm font-semibold text-[#174935]">
              <span>{result.label}</span>
              <span>{(result.confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} layout="vertical" margin={{ top: 8, right: 20, bottom: 8, left: 20 }}>
                  <XAxis type="number" domain={[0, 100]} tickFormatter={(value) => `${value}%`} />
                  <YAxis type="category" dataKey="name" width={72} />
                  <Tooltip formatter={(value) => `${value}%`} />
                  <Bar dataKey="score" radius={[0, 6, 6, 0]}>
                    {chartData.map((entry) => (
                      <Cell
                        key={entry.name}
                        fill={entry.name === result.label ? '#1f5f46' : '#b88342'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
