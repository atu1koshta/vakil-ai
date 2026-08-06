// Active experiment profile chip for the header. Hover shows the full
// config (model, chunking, enrichment) so it's obvious which knobs produced
// what's on screen.
export default function ProfileBadge({ profileInfo }) {
  if (!profileInfo) return null;
  const active = profileInfo.profiles[profileInfo.active];
  if (!active) return null;
  const title = [
    `embedding: ${active.embedding.provider}/${active.embedding.model} (dim ${active.embedding.dim})`,
    `chunking: ${active.chunking.strategy} ${active.chunking.target_tokens}/${active.chunking.overlap_tokens} tokens`,
    `enrich: ${active.indexing.enrich}`,
    `store: ${active.store}`,
    `fingerprint: ${active.fingerprint}`,
  ].join('\n');
  return (
    <span
      title={title}
      className="inline-flex cursor-default items-center gap-1.5 rounded-full border border-indigo-200 bg-indigo-50 px-2.5 py-0.5 text-xs text-indigo-700"
    >
      <span className="h-1.5 w-1.5 rounded-full bg-indigo-500" />
      <span className="font-medium">{profileInfo.active}</span>
      <span className="text-indigo-400">·</span>
      <span>{active.embedding.model}</span>
      <span className="text-indigo-400">·</span>
      <span>{active.chunking.target_tokens}tok</span>
    </span>
  );
}
