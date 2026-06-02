import { Loader2 } from 'lucide-react';

export default function LoadingButton({ children, loading, disabled, ...props }) {
  return (
    <button
      type="button"
      disabled={disabled || loading}
      className="inline-flex min-h-11 items-center justify-center gap-2 rounded-md bg-[#1f5f46] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[#174935] disabled:cursor-not-allowed disabled:bg-[#9baaa0]"
      {...props}
    >
      {loading && <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />}
      {children}
    </button>
  );
}

