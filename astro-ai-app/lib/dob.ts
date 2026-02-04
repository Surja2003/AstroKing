export type DobParseResult = {
  iso: string; // YYYY-MM-DD
  display: string; // DD-MM-YYYY
};

function pad2(n: number) {
  return String(n).padStart(2, '0');
}

export function formatDobIsoToDisplay(isoDob: string): string {
  const t = String(isoDob || '').trim();
  const m = t.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!m) return '';
  const yyyy = m[1];
  const mm = m[2];
  const dd = m[3];
  return `${dd}-${mm}-${yyyy}`;
}

export function parseDobDisplayToIso(displayDob: string): DobParseResult | null {
  const t = String(displayDob || '').trim();
  const m = t.match(/^(\d{2})-(\d{2})-(\d{4})$/);
  if (!m) return null;

  const dd = Number(m[1]);
  const mm = Number(m[2]);
  const yyyy = Number(m[3]);

  if (!Number.isFinite(dd) || !Number.isFinite(mm) || !Number.isFinite(yyyy)) return null;
  if (yyyy < 1900 || yyyy > 2100) return null;
  if (mm < 1 || mm > 12) return null;
  if (dd < 1 || dd > 31) return null;

  // Validate real calendar date.
  const iso = `${String(yyyy)}-${pad2(mm)}-${pad2(dd)}`;
  const d = new Date(`${iso}T00:00:00Z`);
  if (Number.isNaN(d.getTime())) return null;

  // JS Date will roll over invalid dates (e.g., 31 Feb). Check round-trip.
  const rtY = d.getUTCFullYear();
  const rtM = d.getUTCMonth() + 1;
  const rtD = d.getUTCDate();
  if (rtY !== yyyy || rtM !== mm || rtD !== dd) return null;

  return { iso, display: `${pad2(dd)}-${pad2(mm)}-${String(yyyy)}` };
}

// Turns arbitrary user input into a DD-MM-YYYY shaped string while typing.
export function normalizeDobDisplayInput(text: string): string {
  const digits = String(text || '').replace(/\D/g, '').slice(0, 8); // DDMMYYYY
  const d = digits.slice(0, 2);
  const m = digits.slice(2, 4);
  const y = digits.slice(4, 8);

  let out = d;
  if (m) out += `-${m}`;
  if (y) out += `-${y}`;
  return out;
}
