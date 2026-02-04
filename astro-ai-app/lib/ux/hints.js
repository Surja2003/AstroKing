// Smart, non-blocking hints while the user types.

/**
 * @param {string} text
 */
function wordCount(text) {
  const t = String(text || '').trim();
  if (!t) return 0;
  return t.split(/\s+/).filter(Boolean).length;
}

/**
 * @param {string} text
 * @param {'money'|'love'|'mind'|'career'|'general'} intent
 * @param {(key: string, opts?: any) => string} t
 * @returns {string|null}
 */
export function getHint(text, intent, t) {
  const raw = String(text || '');
  const trimmed = raw.trim();
  if (!trimmed) return null;

  const wc = wordCount(trimmed);
  const isVague = trimmed.length < 10 || wc <= 1;
  if (!isVague) return null;

  if (intent === 'money') return t('chat.hints.money');
  if (intent === 'love') return t('chat.hints.love');
  if (intent === 'career') return t('chat.hints.career');
  if (intent === 'mind') return t('chat.hints.mind');

  return t('chat.hints.general');
}
