import { WandSparkles } from 'lucide-react';
import { useState } from 'react';

import { autocomplete } from '../api/api.js';
import ErrorMessage from './ErrorMessage.jsx';
import LoadingButton from './LoadingButton.jsx';
import PresetButtons from './PresetButtons.jsx';

const examples = [
  'it was a dark and stormy night',
  'the creature looked at me',
  'seed text',
];

export default function AutocompleteTab() {
  const [input, setInput] = useState(examples[0]);
  const [method, setMethod] = useState('temperature');
  const [length, setLength] = useState(120);
  const [temperature, setTemperature] = useState(0.7);
  const [beamWidth, setBeamWidth] = useState(5);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const submit = async () => {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const response = await autocomplete({
        input,
        method,
        length,
        temperature,
        beamWidth,
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Autocomplete request failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
      <div className="space-y-4">
        <div>
          <h2 className="text-xl font-semibold">Frankenstein autocomplete</h2>
          <p className="mt-1 text-sm text-[#607064]">Character-level generation from a seed prompt.</p>
        </div>

        <PresetButtons examples={examples} onSelect={setInput} />

        <label className="block">
          <span className="text-sm font-semibold">Seed text</span>
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            rows={6}
            className="mt-2 w-full resize-y rounded-md border border-[#cdd8cc] px-3 py-3 text-sm outline-none transition focus:border-[#1f5f46] focus:ring-2 focus:ring-[#d7eadf]"
          />
        </label>

        <div className="grid gap-3 sm:grid-cols-3">
          {['greedy', 'temperature', 'beam'].map((option) => (
            <button
              key={option}
              type="button"
              onClick={() => setMethod(option)}
              className={`rounded-md border px-3 py-2 text-sm font-semibold capitalize transition ${
                method === option
                  ? 'border-[#1f5f46] bg-[#e6f1ea] text-[#174935]'
                  : 'border-[#cdd8cc] bg-white text-[#334239] hover:border-[#7c9684]'
              }`}
            >
              {option}
            </button>
          ))}
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <label className="block">
            <span className="text-sm font-semibold">Characters: {length}</span>
            <input
              type="range"
              min="10"
              max="500"
              step="10"
              value={length}
              onChange={(event) => setLength(Number(event.target.value))}
              className="mt-3 w-full accent-[#1f5f46]"
            />
          </label>

          {method === 'temperature' && (
            <label className="block">
              <span className="text-sm font-semibold">Temperature: {temperature.toFixed(1)}</span>
              <input
                type="range"
                min="0.1"
                max="2"
                step="0.1"
                value={temperature}
                onChange={(event) => setTemperature(Number(event.target.value))}
                className="mt-3 w-full accent-[#1f5f46]"
              />
            </label>
          )}

          {method === 'beam' && (
            <label className="block">
              <span className="text-sm font-semibold">Beam width: {beamWidth}</span>
              <input
                type="range"
                min="1"
                max="8"
                step="1"
                value={beamWidth}
                onChange={(event) => setBeamWidth(Number(event.target.value))}
                className="mt-3 w-full accent-[#1f5f46]"
              />
            </label>
          )}
        </div>

        <LoadingButton onClick={submit} loading={loading} disabled={!input.trim()}>
          <WandSparkles className="h-4 w-4" aria-hidden="true" />
          Generate
        </LoadingButton>
        <ErrorMessage message={error} />
      </div>

      <div className="min-h-[22rem] rounded-md border border-[#d7ded4] bg-[#f8faf7] p-4">
        <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-[#607064]">Output</h3>
        {!result && (
          <p className="mt-4 text-sm text-[#607064]">Generated text will appear here.</p>
        )}
        {result && (
          <div className="mt-4 space-y-3">
            {result.completions.map((item, index) => (
              <article key={`${item.text}-${index}`} className="rounded-md border border-[#d7ded4] bg-white p-3">
                <div className="mb-2 flex items-center justify-between gap-3 text-xs font-semibold uppercase tracking-[0.1em] text-[#607064]">
                  <span>Candidate {index + 1}</span>
                  {item.score !== null && <span>Score {item.score.toFixed(2)}</span>}
                </div>
                <p className="whitespace-pre-wrap break-words text-sm leading-6">
                  <span className="text-[#607064]">{input.trim()}</span>
                  <span className="font-semibold text-[#17211b]">{item.generated}</span>
                </p>
              </article>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

