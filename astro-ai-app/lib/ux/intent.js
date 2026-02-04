// Intent detection for live typing UX.

export const INTENTS = /** @type {const} */ ({
  money: 'money',
  love: 'love',
  mind: 'mind',
  career: 'career',
  general: 'general',
});

/**
 * @param {string} text
 * @returns {'money'|'love'|'mind'|'career'|'general'}
 */
export function detectIntent(text) {
  const t = String(text || '').toLowerCase();

  if (t.match(/money|finance|income|salary|save|saving|spend|expense|budget|debt|invest/)) return 'money';
  if (t.match(/love|relationship|partner|crush|date|dating|breakup|marriage/)) return 'love';
  if (t.match(/stress|mind|anxiety|overthink|panic|burnout|focus|sleep/)) return 'mind';
  if (t.match(/job|career|work|boss|interview|promotion|resume|exam|study|college|school/)) return 'career';

  return 'general';
}

/**
 * @param {'money'|'love'|'mind'|'career'|'general'} intent
 */
export function intentEmoji(intent) {
  switch (intent) {
    case 'money':
      return '💰';
    case 'love':
      return '❤️';
    case 'mind':
      return '🧠';
    case 'career':
      return '🎯';
    default:
      return '✨';
  }
}
