import { getItem, setItem } from '../storage';

const KEY_PREFIX = 'ak:feedback:';

/**
 * @param {string} userKey
 */
function key(userKey) {
  return `${KEY_PREFIX}${userKey}`;
}

function todayKey() {
  const d = new Date();
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  return `${yyyy}-${mm}-${dd}`;
}

/**
 * @typedef {{ d: string; v: 'positive'|'neutral'|'negative' }} FeedbackEntry
 */

/**
 * @param {string} userKey
 * @returns {Promise<FeedbackEntry[]>}
 */
export async function getFeedbackHistory(userKey) {
  try {
    const raw = await getItem(key(userKey));
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

/**
 * @param {string} userKey
 * @param {'positive'|'neutral'|'negative'} value
 */
export async function saveFeedback(userKey, value) {
  if (!userKey) return;
  const entry = /** @type {FeedbackEntry} */ ({ d: todayKey(), v: value });
  const prev = await getFeedbackHistory(userKey);
  const next = [entry, ...prev].slice(0, 60);
  try {
    await setItem(key(userKey), JSON.stringify(next));
  } catch {
    // ignore
  }
}

/**
 * @param {FeedbackEntry[]} entries
 */
function score(entries) {
  let s = 0;
  for (const e of entries) {
    if (e?.v === 'positive') s += 1;
    if (e?.v === 'negative') s -= 1;
  }
  return s;
}

/**
 * @param {string} userKey
 * @returns {Promise<'supportive'|'balanced'|'motivational'>}
 */
export async function getToneHint(userKey) {
  const hist = await getFeedbackHistory(userKey);
  const recent = hist.slice(0, 14);
  const s = score(recent);
  if (s <= -2) return 'supportive';
  if (s >= 2) return 'motivational';
  return 'balanced';
}
