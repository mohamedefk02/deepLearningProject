export default function ErrorMessage({ message }) {
  if (!message) return null;
  return (
    <div className="rounded-md border border-[#e2b8b8] bg-[#fff1f1] px-3 py-2 text-sm font-medium text-[#8f2525]">
      {message}
    </div>
  );
}

