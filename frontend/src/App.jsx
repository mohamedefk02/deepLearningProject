import { Languages, Newspaper, Sparkles } from 'lucide-react';
import { useEffect, useState } from 'react';

import AutocompleteTab from './components/AutocompleteTab.jsx';
import ClassificationTab from './components/ClassificationTab.jsx';
import TranslationTab from './components/TranslationTab.jsx';
import { health } from './api/api.js';

const tabs = [
  { id: 'autocomplete', label: 'Autocomplete', icon: Sparkles },
  { id: 'translation', label: 'Translation', icon: Languages },
  { id: 'classification', label: 'News', icon: Newspaper },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('autocomplete');
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    health()
      .then(() => setApiStatus('ready'))
      .catch(() => setApiStatus('offline'));
  }, []);

  return (
    <main className="min-h-screen bg-[#f6f7f4] text-[#17211b]">
      <div className="mx-auto flex min-h-screen w-full max-w-6xl flex-col px-4 py-5 sm:px-6 lg:px-8">
        <header className="mb-4 flex flex-col gap-3 border-b border-[#d7ded4] pb-4 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.12em] text-[#607064]">
              LSTM model demo
            </p>
            <h1 className="mt-1 text-3xl font-semibold text-[#17211b] sm:text-4xl">
              Deep Learning Suite
            </h1>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                apiStatus === 'ready'
                  ? 'bg-[#2f8f5b]'
                  : apiStatus === 'offline'
                    ? 'bg-[#bf3b3b]'
                    : 'bg-[#c78a1d]'
              }`}
            />
            <span className="font-medium">
              API {apiStatus === 'ready' ? 'ready' : apiStatus === 'offline' ? 'offline' : 'checking'}
            </span>
          </div>
        </header>

        <nav className="mb-5 flex flex-wrap gap-2" aria-label="Model tabs">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const selected = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`inline-flex min-h-11 items-center gap-2 rounded-md border px-4 py-2 text-sm font-semibold transition ${
                  selected
                    ? 'border-[#1f5f46] bg-[#1f5f46] text-white'
                    : 'border-[#ccd5cb] bg-white text-[#26352c] hover:border-[#7c9684]'
                }`}
              >
                <Icon className="h-4 w-4" aria-hidden="true" />
                {tab.label}
              </button>
            );
          })}
        </nav>

        <section className="flex-1 rounded-lg border border-[#d7ded4] bg-white p-4 shadow-sm sm:p-6">
          {activeTab === 'autocomplete' && <AutocompleteTab />}
          {activeTab === 'translation' && <TranslationTab />}
          {activeTab === 'classification' && <ClassificationTab />}
        </section>
      </div>
    </main>
  );
}

