export default function PresetButtons({ examples, onSelect }) {
  return (
    <div className="flex flex-wrap gap-2">
      {examples.map((example) => (
        <button
          key={example}
          type="button"
          onClick={() => onSelect(example)}
          className="rounded-md border border-[#cdd8cc] bg-[#f8faf7] px-3 py-2 text-left text-sm font-medium text-[#334239] transition hover:border-[#7c9684] hover:bg-[#eef4ef]"
        >
          {example}
        </button>
      ))}
    </div>
  );
}

