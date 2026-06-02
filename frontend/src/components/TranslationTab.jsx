import { Languages } from 'lucide-react';
import { useEffect, useState } from 'react';

import { translate } from '../api/api.js';
import ErrorMessage from './ErrorMessage.jsx';
import LoadingButton from './LoadingButton.jsx';
import PresetButtons from './PresetButtons.jsx';

const examples = [
  'this book is interesting',
  'where is the library',
  'the weather is beautiful today',
];

export default function TranslationTab() {
  const [input, setInput] = useState(examples[0]);
  const [result, setResult] = useState(null);
  const [visibleWords, setVisibleWords] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!result?.words?.length) {
      setVisibleWords([]);
      return undefined;
    }

    setVisibleWords([]);
    let index = 0;
    const interval = window.setInterval(() => {
      index += 1;
      setVisibleWords(result.words.slice(0, index));
      if (index >= result.words.length) {
        window.clearInterval(interval);
      }
    }, 180);

    return () => window.clearInterval(interval);
  }, [result]);

  const submit = async () => {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const response = await translate(input);
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Translation request failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
      <div className="space-y-4">
        <div>
          <h2 className="text-xl font-semibold">English to French translation</h2>
          <p className="mt-1 text-sm text-[#607064]">Seq2Seq LSTM decoder output animated word by word.</p>
        </div>

        <PresetButtons examples={examples} onSelect={setInput} />

        <label className="block">
          <span className="text-sm font-semibold">English sentence</span>
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            className="mt-2 w-full rounded-md border border-[#cdd8cc] px-3 py-3 text-sm outline-none transition focus:border-[#1f5f46] focus:ring-2 focus:ring-[#d7eadf]"
          />
        </label>

        <LoadingButton onClick={submit} loading={loading} disabled={!input.trim()}>
          <Languages className="h-4 w-4" aria-hidden="true" />
          Translate
        </LoadingButton>
        <ErrorMessage message={error} />
      </div>

      <div className="min-h-[18rem] rounded-md border border-[#d7ded4] bg-[#f8faf7] p-4">
        <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-[#607064]">Output</h3>
        {!result && <p className="mt-4 text-sm text-[#607064]">French translation will appear here.</p>}
        {result && (
          <div className="mt-5">
            <p className="text-sm font-semibold text-[#607064]">Original</p>
            <p className="mt-1 break-words text-base text-[#17211b]">{input}</p>
            <p className="mt-6 text-sm font-semibold text-[#607064]">French</p>
            <p className="mt-2 min-h-12 break-words text-2xl font-semibold leading-9 text-[#1f5f46]">
              {visibleWords.join(' ')}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
