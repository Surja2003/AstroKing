import { getItem, setItem } from '../storage';

const KEY_PREFIX = 'ak:recentTopics:';

/**
 * @param {string} userKey
 */
function key(userKey) {
  return `${KEY_PREFIX}${userKey}`;
}

/**
 * @param {string} userKey
 * @returns {Promise<string[]>}
 */
export async function getRecentTopics(userKey) {
  try {
    const raw = await getItem(key(userKey));
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((x) => typeof x === 'string') : [];
  } catch {
    return [];
  }
}

/**
 * @param {string} userKey
 * @param {'money'|'love'|'mind'|'career'|'general'} intent
 * @returns {Promise<string[]>} updated
 */
export async function updateRecentTopics(userKey, intent) {
  if (!userKey) return [];
  if (!intent || intent === 'general') return getRecentTopics(userKey);

  const prev = await getRecentTopics(userKey);
  const next = [intent, ...prev.filter((x) => x !== intent)].slice(0, 3);
  try {
    await setItem(key(userKey), JSON.stringify(next));
  } catch {
    // ignore
  }
  return next;
}
